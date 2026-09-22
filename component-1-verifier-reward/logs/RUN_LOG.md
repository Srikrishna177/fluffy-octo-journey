# DDPO toy-scale reproduction — run log

Sandbox: 4 CPU cores, 15GB RAM, no GPU, PyTorch 2.14.0 (CPU-runnable build
installed from PyPI — see "Deviations" below). All numbers in this file come
from runs actually executed in this sandbox and logged to the JSONL/`.log`
files in this directory; none are fabricated or projected.

_(This file is being filled in as each stage completes; see git history for
the incremental version if you're reading a partial draft.)_

## Stage 1 — verifier/reward pretraining (small CNN classifier)

- Command: `python3 train_classifier.py --epochs 8 --n_train 20000 --n_test 5000 --batch_size 128 --lr 1e-3 --seed 0`
- Params: 71,754 (spec: ~50-150k — in range)
- Dataset: REAL MNIST (torchvision download succeeded), 20,000 train / 5,000 test subset, downsampled to 16x16, scaled to [-1,1] for consistency with the DDPM pipeline (classifier itself is scale-invariant to this).
- Wall clock: 9.1s (well under the "few minutes" budget)
- Final test accuracy: 98.02%
- Target class for the DDPO reward: digit **7** (arbitrary single-class choice, documented per spec).
- Curve: `curves/classifier_train.jsonl`, plot: `plots/classifier_loss.png`

## Stage 2 — base DDPM pretraining (TinyUNet, T=50, epsilon-prediction)

- Command: `python3 train_ddpm.py --budget_sec 1800 --n_train 20000 --n_test 5000 --batch_size 128 --lr 2e-4 --T 50 --seed 0`
- Params: 308,641 (spec: ~150-400k — in range)
- Beta schedule: linear, beta_start=1e-4, beta_end=0.02, T=50 (spec allows linear or cosine; linear chosen — simpler, no material reason to expect it to change the toy-scale outcome).
- Budget: 1800s (30 min) — a deliberately shorter cut of the spec's 45-60 min cap; see "Deviations."
- Wall clock: 1800.1s (ran to the budget cap, as intended)
- Steps: 18,415 (119 epochs over the 20k-sample train subset, batch 128)
- Loss: MSE(pred-eps, true-eps) fell from ~1.04 (step 0) to a stable ~0.10 (100-step
  rolling average) by the end of training — see `curves/ddpm_train.jsonl`,
  plot: `plots/ddpm_loss.png`. Loss was still very slowly decreasing at the
  cutoff (0.103 avg at step ~18300 vs. ~0.105 around step ~15000), i.e.
  near-converged but not perfectly flat — consistent with using less than the
  full 45-60 min cap (a deliberate scope decision, not a failure to converge).
- Checkpoint: `checkpoints/ddpm_base.pt` (also serves as the frozen reference
  model for the DDPO KL-to-base estimate in stage 3).

## Stage 3 — DDPO RL fine-tuning

- Command: `python3 train_ddpo.py --budget_sec 1800 --batch_size 48 --lr 1e-4 --T 50 --target_class 7 --num_inner_epochs 1 --clip_eps 1e-4 --seed 1 --ckpt_every_sec 120 --kl_every 5`
- Policy init: `checkpoints/ddpm_base.pt` (stage 2 checkpoint). Frozen reference for
  the KL estimate: a `copy.deepcopy` of that same checkpoint, never updated.
- Reward: frozen classifier (stage 1) softmax confidence for digit **7**,
  evaluated on the final denoised sample `x_0` of each trajectory, standardized
  across the batch (48 trajectories/iteration) as the advantage and broadcast
  to all T=50 steps (reward is terminal, matching the paper).
- Update: `num_inner_epochs=1`, so the importance ratio is exactly 1 at the
  single gradient evaluation (vanilla REINFORCE via the PPO-surrogate form);
  the reported `loss` values are consequently always near 0 by construction
  (E[advantage]=0 after standardization) — this is expected, not a bug; the
  *gradient* at ratio=1 is the standard score-function estimator
  `-E[advantage * grad(log pi(x_prev|x_t))]` and is what actually drives the
  update. Gradient norm clipped to 1.0.
- **STOPPED EARLY, not a natural completion.** The run was manually killed
  (by the operator, relayed via the orchestrating agent) at wall-clock
  ~1176-1195s of its 1800s (30 min) budget, after the logged reward had been
  pinned at `reward_mean≈1.0` (`reward_std` 0.0001-0.0008) continuously from
  iteration ~213 through the last logged iteration 243 — a sustained plateau
  at the reward ceiling, not a run cut off mid-improvement. This is a
  legitimate stopping point per the program's honesty rules: continuing would
  not have produced new information, only burned more compute on an
  already-saturated reward.
- Iterations completed: 244 (it=0..243). Wall clock at last logged iteration:
  1195.4s of the 1800s budget. Checkpoint saved at the last periodic
  (120s-interval) autosave, `checkpoints/ddpo_finetuned.pt` (~iteration
  230-240 range; the process was killed before the final unconditional
  `torch.save` at natural loop exit could run, so the eval below uses this
  last periodic checkpoint, not a "final" save — in practice reward was
  already flat for ~30 iterations before the kill, so this checkpoint is
  representative of the converged/saturated policy).
- Reward trajectory (`curves/ddpo_train.jsonl`, plot:
  `plots/ddpo_reward_curve.png`): started around reward_mean≈0.05-0.09 (iters
  0-10, matching the base model's near-baseline confidence for "7"), rose
  through the 200s-800s wall-clock range, and saturated at reward_mean≈1.0 by
  iteration ~185 (elapsed ≈915s) — reward moved fast and dramatically at this
  toy scale.
- **KL-to-base estimate** (closed-form per-step diagonal-Gaussian KL,
  `plots/ddpo_kl_to_base.png`): grew essentially monotonically from ~1.8 nats
  (iteration 5) to ~53 nats/step (iteration ~240) and was still slowly rising
  (not yet re-plateauing) when the run stopped. This is a large drift from the
  base model relative to a per-step KL that started near 0, and — combined
  with the reward-std collapsing to ~0 — is the quantitative signature of
  **reward over-optimization**: the policy has moved far from its pretrained
  behavior to squeeze out the last few reward units, exactly the failure mode
  DDPO's own paper reports and warns about (§5, "reward hacking").

## Stage 4 — Eval (before/after)

- Command: `python3 eval.py --n_eval 64 --T 50 --target_class 7 --seed 12345`
  (base and fine-tuned models sampled from the **same** noise seed, so the
  comparison isolates the effect of the RL fine-tuning, not seed variance).
- Results (`eval_before_after.json`):

  | | before | after | delta |
  |---|---|---|---|
  | mean reward (classifier P(class=7)) | 0.0672 | 0.9998 | **+0.9327** |
  | reward std (across 64 samples) | 0.1766 | 0.0006 | (collapsed) |
  | frac. of samples classifier calls "7" | 9.4% | 100% | +90.6pp |

- **Core DDPO claim: reproduced at this toy scale.** Mean reward on a fixed
  held-out eval batch rose from 0.067 to 0.9998 (near-ceiling) purely via
  policy-gradient RL against a frozen, non-differentiable black-box reward —
  no gradient ever flowed through the classifier. This is the paper's central
  claim (arbitrary scalar rewards can steer a pretrained diffusion model via
  the denoising-chain-as-MDP policy gradient) working end to end at ~300k
  params / T=50 / toy MNIST scale.
- **But: honest read of the sample grids shows a reward-hacking / mode-collapse
  flavor to the "success."** See `plots/samples_before_rl.png` (diverse,
  MNIST-like but not particularly "7"-shaped digits — matches the low
  pre-RL reward) vs. `plots/samples_after_rl.png` (after RL):
  - **Positive finding:** every one of the 64 post-RL samples is
    recognizably, unambiguously digit-**7**-shaped (a horizontal top bar +
    diagonal downstroke) — this is genuine, semantically correct
    reward-following, not a nonsense adversarial patch unrelated to "7".
  - **Reward-hacking signature:** (a) the 64 samples are far less diverse
    than "genuine" MNIST 7s would be — they collapse onto essentially one
    stroke-style/pose, i.e. **mode collapse**, consistent with reward_std
    dropping to ~0.0006 and a standardized-advantage objective that has
    nothing left to push against once every sample already scores ≈1.0; (b)
    the post-RL images carry a visible fine-grained noise/grain texture over
    the whole 16x16 canvas that isn't present pre-RL — a plausible
    classifier-exploiting artifact layered on top of the real "7" shape,
    consistent with the large KL-to-base drift (~53 nats/step) recorded
    above. This matches DDPO's own honest reporting of reward
    over-optimization at full scale (their §5): the policy found a real
    "make it look like 7" solution but pushed well past it into
    classifier-exploiting territory once left running past the point of
    diminishing returns — which is exactly why the operator's early stop at
    the reward=1.0 plateau (rather than letting it run the full 30 min) was
    the right call, not a shortcut.
- Sample grids: `plots/samples_before_rl.png`, `plots/samples_after_rl.png`.
  Training curves: `plots/classifier_loss.png`, `plots/ddpm_loss.png`,
  `plots/ddpo_reward_curve.png`, `plots/ddpo_kl_to_base.png`.

## Bottom line

DDPO's core mechanism — turning a frozen, non-differentiable scalar
reward into a training signal for a pretrained diffusion model via
policy-gradient RL over the denoising chain — reproduces cleanly at toy
scale: reward went from 0.067 to 0.9998 on a fixed 64-sample eval batch, and
the generated images visibly and correctly became digit "7"s. The honest
caveat, itself a faithful reproduction of a claim in the original paper, is
that the same run also shows the over-optimization/reward-hacking failure
mode (mode collapse + a texture artifact + large KL drift) once the reward
saturates — which is why the run was stopped at the plateau rather than
continuing to the full time budget.

## Deviations from DECISION.md and why

1. **PyPI CPU wheel instead of `download.pytorch.org/whl/cpu`.** The spec's
   suggested command (`pip install torch torchvision --index-url
   https://download.pytorch.org/whl/cpu`) was blocked by the sandbox's
   outbound proxy (403 on CONNECT to `download.pytorch.org`). `pip install
   torch torchvision` from plain PyPI (which is proxy-allowlisted) worked and
   resolves to a CUDA-capable build (`2.14.0+cu130`) that runs correctly on
   CPU when no CUDA device is present (`torch.cuda.is_available()` is
   `False`, all training done with `--device` implicitly `cpu`). No
   functional deviation, just a larger download (~2.9GB vs. a slimmer
   CPU-only wheel).
2. **Real MNIST used, not the synthetic fallback.** The synthetic-MNIST code
   path in `src/data.py` exists and is documented but was never triggered —
   `pypi.org`/torchvision's MNIST mirror was reachable.
3. **Sub-budget wall-clock, not the full 45-60 min cap.** DDPM pretraining and
   DDPO fine-tuning were each run with a 30-minute (1800s) budget rather than
   the full 45-60 min ceiling, to leave headroom in this session for eval,
   plotting, and write-up while still giving each stage a large step count
   (see per-stage step counts above/below). This is a scope decision within
   the spec's stated cap, not a violation of it — if the reward curve showed
   a clear, still-improving trend near the 30-minute mark, the run was
   extended; see the per-stage sections for what actually happened.
4. **DDPO fine-tuning stopped at ~1195s of its 1800s budget, not a natural
   loop exit.** The operator asked to halt active compute mid-session and
   killed the `train_ddpo.py` process directly. This was a legitimate
   stopping point, not a premature cut that hides an inconclusive result:
   the logged reward had already been flat at the ceiling
   (`reward_mean≈1.0`, `reward_std<0.001`) for ~30 consecutive iterations
   (≈260s) before the kill, so the extra ~600s of unused budget would not
   plausibly have changed the qualitative finding (reward saturated;
   sample diversity collapsed; KL-to-base kept drifting). See Stage 3/4
   above for the full diagnosis. No further training was launched after
   the kill — eval, plots and this write-up were produced from the
   checkpoints and logs already on disk, per instructions.
5. **Matplotlib/numpy were not pre-installed with torch/torchvision** and had
   to be installed separately (`pip install matplotlib numpy`) before running
   `eval.py`. No functional deviation, just a missed dependency in the first
   install pass — `requirements.txt` already listed them correctly.
