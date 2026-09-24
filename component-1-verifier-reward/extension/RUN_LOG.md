# Component 1 extension — internal-verifier DDPO run log

Sandbox: 4 CPU cores, 15GB RAM, no GPU (same as the reproduction). All
numbers in this file come from runs actually executed in this sandbox and
logged to the JSONL/`.log`/`.json` files in `logs/`; none are fabricated or
projected.

## Recap: what changed vs. the reproduction

Everything held constant except the reward source: same frozen pretrained
DDPM (`repro/checkpoints/ddpm_base.pt`, never retrained), same DDPO
Algorithm 1 implementation (`extension/src/train_ddpo_internal.py` is
line-for-line identical to `repro/src/train_ddpo.py` apart from the reward
call), same hyperparameters (T=50, batch_size=48, lr=1e-4,
num_inner_epochs=1, clip_eps=1e-4, target class 7, seed=1), same time-box
philosophy (cap at 1200s, stop early at a sustained reward plateau exactly
as the reproduction did).

## Stage 1 — internal verifier: linear probe on frozen DDPM features

- Command: `python3 train_probe.py --epochs 8 --n_train 20000 --n_test 5000 --batch_size 128 --lr 1e-3 --seed 0`
- Feature source: `repro/checkpoints/ddpm_base.pt`'s `mid` ResBlock output
  (bottleneck, 64x8x8=4096-dim), read via a forward hook attached from
  outside the model (`extension/src/probe.py:FeatureExtractor`) —
  `repro/src/unet.py` itself is untouched, matching the "add a hook, don't
  rewrite the model" instruction. Input to the frozen UNet: real
  MNIST-16x16 images passed through `q_sample` at the fixed near-zero
  timestep t=0.
- Probe: a single `nn.Linear(4096, 10)`, 40,970 params.
- Dataset: same real-MNIST pipeline as the reproduction (`repro/src/data.py`),
  20,000 train / 5,000 test.
- Wall clock: 40.7s.
- Final test accuracy: **97.62%** (vs. the external classifier's 98.02% —
  the frozen DDPM's own bottleneck features are nearly as linearly
  separable by digit identity as a CNN trained end-to-end for
  classification, even though the DDPM was never given a classification
  objective).
- Curve: `logs/curves/probe_train.jsonl`, plot: `logs/plots/probe_train_loss.png`.

## Stage 2 — DDPO treatment run (reward = internal-probe confidence)

- Command: `python3 train_ddpo_internal.py --budget_sec 1200 --batch_size 48 --lr 1e-4 --T 50 --target_class 7 --num_inner_epochs 1 --clip_eps 1e-4 --seed 1 --ckpt_every_sec 120 --kl_every 5`
- Policy init: `repro/checkpoints/ddpm_base.pt`, identical to the
  reproduction. Frozen reference for KL: a `copy.deepcopy` of that same
  checkpoint. Frozen feature-extractor backbone for the reward: a
  *second*, separately-deepcopied frozen copy of the same base checkpoint
  (never updated), feeding the frozen probe.
- Reward: internal-probe softmax confidence for digit 7, computed from the
  frozen base-UNet's bottleneck feature at t=0 — the internal-verifier
  analogue of `classifier.reward_fn`, standardized across the batch (48
  trajectories/iteration) as the advantage, broadcast to all T=50 steps.
- **STOPPED EARLY at a sustained plateau, same practice as the
  reproduction.** Reward climbed from ~0.01-0.09 (iters 0-10) through
  elapsed 100-400s, first touched >0.99 at iteration 89 (elapsed 400.0s),
  and was pinned at `reward_mean` 0.999-1.000 with `reward_std<0.002`
  continuously from iteration ~92 through the last logged iteration 114
  (elapsed 505.5s) — over 20 iterations flat at the reward ceiling. The
  process was killed at this point (505.5s of the 1200s budget) because
  continuing would not have produced new information about the *training*
  reward, matching the reproduction's own stopping rationale. **Notably,
  the internal-reward run reached its own reward ceiling roughly 2.3x
  faster in wall-clock than the reproduction did** (~400s vs. ~915s to
  first sustained saturation).
- Iterations completed: 115 (it=0..114). Checkpoint saved at the last
  periodic (120s-interval) autosave, `checkpoints/ddpo_internal_finetuned.pt`
  (~elapsed 484s, iteration ~110 range) — same caveat as the reproduction:
  the process was killed before the final unconditional save, so this is
  the last periodic checkpoint, not a "natural" loop exit; reward was
  already flat for >20 iterations before the kill, so it is representative
  of the saturated policy.
- Reward trajectory: `logs/curves/ddpo_internal_train.jsonl`, plot:
  `logs/plots/ddpo_internal_reward_curve.png`.
- **KL-to-base estimate** (`logs/plots/ddpo_internal_kl_to_base.png`): grew
  from ~1.3 nats (iteration 0) to a peak of ~18.8 nats around iteration 55
  (elapsed 255.6s, still pre-saturation), then **plateaued and mildly
  declined/oscillated** (18.8 -> 15.9 -> 15.4 -> 12.4 -> 10.4 -> 10.3 nats)
  from iteration ~75 through the last logged iteration 110 (elapsed 488.8s)
  — i.e. once the training reward saturated, KL-to-base did **not** keep
  climbing the way it did in the reproduction (see comparison table below).

## Stage 3 — independent-judge eval (external frozen classifier)

- Command: `python3 eval_internal.py --n_eval 64 --target_class 7 --seed 12345`
- Same protocol as the reproduction's `eval.py`: same n=64, same seed
  12345, base and fine-tuned models sampled from the same noise seed. The
  "before" numbers are identical to the reproduction's own eval (same base
  checkpoint, same seed) — reported here for completeness, not re-derived.
- Judge: the ORIGINAL frozen external classifier
  (`repro/checkpoints/classifier.pt`), never used anywhere in the
  internal-reward training loop — a true independent judge.
- Results (`logs/eval_external_judge.json`):

  | | before | after (internal-reward treatment) |
  |---|---|---|
  | external-judge mean reward | 0.0672 | **0.9594** |
  | external-judge reward std | 0.1766 | **0.1268** |
  | external-judge frac. argmax==7 | 9.4% | **98.4%** |
  | internal-probe's own opinion (mean) | 0.0191 | 0.99996 (std 5.4e-5) |

## Comparison table: reproduction (external reward) vs. extension (internal reward)

All numbers below are read directly from the two arms' JSONL/JSON logs
(`extension/src/compare.py` recomputes this table from the raw files).

| metric | reproduction (external reward) | extension (internal reward) |
|---|---|---|
| final training-reward (own reward source, last logged iter) | 0.9999 | 0.9999 |
| final training-reward-std (own reward source, last logged iter) | 0.0003 | 0.0001 |
| iterations completed / wall clock to stop | 244 / 1195.4s | 115 / 505.5s |
| wall-clock to first sustained saturation | ~915s | ~400s |
| external-judge reward_mean (post-RL, n=64, seed=12345) | 0.9998 | 0.9594 |
| external-judge reward_std (post-RL) | 0.0006 | 0.1268 |
| external-judge frac_argmax_is_target (post-RL) | 100% | 98.4% |
| KL-to-base at ~120s elapsed | 8.41 (it=25) | 4.79 (it=25) |
| KL-to-base at ~300s elapsed | 24.25 (it=60) | 16.62 (it=65) |
| KL-to-base peak during run | 53.38 nats, still rising at kill (elapsed 1180s) | 18.77 nats (elapsed 255.6s), then **declines/plateaus** to 10.3 nats by elapsed 488.8s |
| KL-to-base trajectory shape | **monotonic climb**, never re-plateaus | rises then **plateaus/oscillates downward** once reward saturates |

## Qualitative check: sample grids (the number that isn't in the table)

`logs/plots/samples_before_rl.png` vs. `logs/plots/samples_after_rl.png`
(internal-reward treatment), compared against the reproduction's own
`../logs/plots/samples_after_rl.png` (external-reward baseline):

- **Baseline (external reward), per the reproduction's RUN_LOG:** every one
  of the 64 post-RL samples is recognizably, unambiguously digit-**7**-shaped
  (a horizontal top bar + diagonal downstroke), with a visible fine-grained
  noise texture layered on top (a classifier-exploiting artifact) and heavy
  mode collapse onto one stroke style.
- **Extension (internal-probe reward), this run:** the 64 post-RL samples
  collapse onto a **cruder pattern** — a bright horizontal bar occupying
  roughly the top third of the canvas with faint scattered noise below, and
  **no consistent diagonal downstroke**. Visually this reads far less
  convincingly as a handwritten "7" to a human than the baseline's samples
  do, even though it fools the internal probe almost perfectly (reward
  0.99996, std ~5e-5) and fools the *external* classifier reasonably well
  too (mean 0.959, 98.4% argmax==7).
- This is the most important honest finding of this extension, and it is
  **not visible in the external-judge scalar summary alone**: the
  internal-reward run's "success," judged only by the external-classifier
  numbers, looks almost as strong as the baseline's (0.959 vs. 0.9998) and
  even *more diverse* (std 0.127 vs. 0.0006) and *lower-drift* (KL
  plateaus at ~10-19 nats vs. climbing past 53). But the sample grids show
  the policy found a **cheaper, less semantically genuine shortcut** — a
  simplified bright-bar pattern — that happens to transfer to fool the
  independent external classifier fairly well, rather than converging on
  an actual "7" shape the way the baseline did.

## Which hypothesis this supports

`extension/DECISION.md` posed two candidate hypotheses with no strong
prior: internal-verifier reward could be **easier to hack** (coarser probe
features let the policy get away with a less-real solution) or **harder to
hack** (features tied to the generator's own representation resist gaming).

**The data are genuinely mixed, and reporting them as a clean win for
either hypothesis would be dishonest:**

- **Evidence for "easier to hack" (hypothesis 1):** the qualitative sample
  grids are the most damning evidence here. The internal-reward run
  converged to a visually cruder, less recognizably-"7" degenerate pattern
  (a plain horizontal bar) than the baseline's genuine (if
  texture-corrupted) 7-shapes — exactly the failure mode hypothesis 1
  predicted: a coarser, lower-capacity linear probe on 4096-dim frozen
  features imposes a weaker constraint on "what counts as confidently
  class 7" than a dedicated pixel-level CNN does, so the policy found a
  cheaper shortcut that still happens to transfer to the external judge.
- **Evidence that looks like "harder to hack" (hypothesis 2), but is
  probably a confound:** the internal-reward run shows a much healthier
  *optimization trajectory* by the numbers — KL-to-base plateaus at ~10-19
  nats instead of climbing past 53, and it does so while reaching its own
  reward ceiling ~2.3x faster. This is very likely an artifact of the
  reward landscape shape, not evidence the internal verifier is
  intrinsically harder to fool: once the policy finds the horizontal-bar
  shortcut, the internal probe's confidence is *already* pinned near 1.0
  for essentially every sample, so the standardized-advantage objective
  (which amplifies whatever residual reward variance is left, however
  small) has very little further gradient signal to keep pushing the
  policy away from its base weights. The baseline's classifier, by
  contrast, kept offering a thin residual gradient (rising confidence in
  the fourth decimal place) that the standardized advantage kept amplifying
  into continued policy drift long after the reward looked "saturated" on
  paper. In other words: the extension's lower KL drift is best read as
  "it stopped moving because it found a cheap local optimum fast," not as
  "the reward source resists being gamed."
- **A confound we did not fully control:** the treatment's post-saturation
  window (elapsed ~400-505s, ~105s at the ceiling) is much shorter than the
  baseline's (elapsed ~915-1195s, ~280s at the ceiling), because the
  treatment was stopped as soon as it hit a comparably sustained plateau.
  We cannot rule out that letting the treatment run for the same ~280s
  *at* saturation (rather than ~105s) would have eventually shown the same
  kind of continued KL creep the baseline did. This experiment does not
  answer that question; a controlled version would match wall-clock time
  spent at saturation, not total wall-clock, between arms.

**Bottom line:** on the visual/semantic evidence (the sample grids), this
run supports **hypothesis 1 (internal-verifier reward is easier to hack)**
— the policy exploited the coarser linear probe with a less genuine
degenerate solution than the external CNN allowed. On the purely numeric
KL/reward-std evidence, the run superficially resembles hypothesis 2, but
we judge that read to be a confound of how much gradient signal remained
after each reward source's own saturation point, not a real difference in
robustness. We report both reads plainly rather than picking the one that
tells a cleaner story.

## Deviations from extension/DECISION.md and why

1. **Time-box realized as ~505.5s of the ~1200s cap, not the full cap.**
   Per the spec ("if the internal-verifier reward doesn't move ... that's a
   logged negative result, not a reason to extend"), the converse also
   applies at this toy scale: once the reward has been pinned at the
   ceiling with near-zero std for >20 consecutive iterations, continuing
   only burns compute without changing the qualitative finding — the exact
   same judgment call the reproduction made at ~1195s of its 1800s budget.
   This is a scope decision, not a shortfall; see "confound" note above for
   the one place this decision limits what the run can tell us (matched
   time-at-saturation).
2. **Feature extractor implemented via an external `forward_hook`, not a
   new method on `TinyUNet`.** `repro/src/unet.py` is completely untouched;
   `extension/src/probe.py:FeatureExtractor` attaches
   `register_forward_hook` to the `mid` submodule from outside the class.
   This satisfies the instruction to "add a forward-hook or a method... not
   rewrite the model" via the hook option, while leaving the already-
   completed reproduction's code byte-for-byte unmodified.
3. **The reward's frozen feature-extractor backbone is a separate
   `deepcopy` of the base checkpoint from the KL reference network**,
   rather than reusing one object for both roles. This is a memory/clarity
   choice (both are frozen, never-updated copies of the identical
   checkpoint; using two objects avoids any risk of the forward hook's
   cached activation being read at the wrong point in a shared call
   sequence) and has no effect on the results.
4. No other deviations: same T=50, batch_size=48, lr=1e-4,
   num_inner_epochs=1, clip_eps=1e-4, target class 7, same MNIST-16x16 data
   pipeline, same eval protocol (n=64, seed=12345) as both the reproduction
   and `extension/DECISION.md` specify.
