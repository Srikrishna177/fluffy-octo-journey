# Component 3 extension — on-policy training states — run log

Sandbox: 4 CPU cores, 15GB RAM, no GPU, PyTorch (CPU build, same environment
as the reproduction). All numbers below come from a run actually executed in
this sandbox and logged to `logs/results.json` /
`logs/curves/distill_train_onpolicy.jsonl`; none are fabricated or
projected.

## Command

```bash
cd component-3-adaptive-distill/extension/src
python3 distill_onpolicy.py \
  --pool_size 4096 --pool_batch 512 --pool_seed 999 \
  --batch_size 64 --lr 2e-4 --round_budget_sec 300 --round_max_steps 4000 --train_seed 101 \
  --n_eval 64 --eval_seed 12345
python3 plots_ext.py
```

Launched as a background process (per the time-box instructions), polled
periodically via log tail rather than blocking a single call.

## Hyperparameters — matched exactly to the reproduction's round-1 (32→16) run

| | reproduction round 1 (data-marginal) | this extension (on-policy) |
|---|---|---|
| student init | deep copy of frozen teacher | deep copy of frozen teacher (same) |
| architecture | TinyUNet, 308,641 params | same |
| batch_size | 64 | 64 |
| lr | 2e-4 | 2e-4 |
| round_budget_sec | 300 | 300 |
| round_max_steps | 4000 | 4000 |
| training seed | 101 (100 + round_idx 1) | 101 |
| **training input states `z_ti`** | `q_sample(real_mnist_x0, t_i, fresh_eps)` (data marginal) | **on-policy replay pool**: states actually visited by the teacher's own 32-step DDIM rollout from pure noise |
| regression target | `teacher_two_step_target` + `invert_x_target` (paper Alg. 2) | identical functions, imported unchanged from `repro/src/distill.py` |

Only the source of `z_ti` differs. Verified by direct code reuse
(`from distill import eval_arm, invert_x_target, teacher_two_step_target`),
not a reimplementation.

## On-policy pool generation

Rolled the frozen teacher out from pure noise on its own 32-step DDIM
schedule 4,096 times (8 rollout batches of 512), recording the state
actually visited at each of the 16 pair-start timesteps (`t_i = schedule[2*p]`
for `p = 0..15`) along each rollout — a genuine "replay buffer" of on-policy
`(state, timestep)` pairs, not a data-marginal proxy. This is cheap on this
CPU sandbox (a 32-step teacher rollout batch of 512 takes ~5s), so 4,096 pool
entries per pair (comparable in order of magnitude to the reproduction's
10,000-image MNIST pool) took:

**Pool generation wall-clock: 44.3s.**

## Training run

**Training wall-clock: 300.3s** (hit the time budget, same as the
reproduction's round 1), **1,059 gradient steps** (vs. the reproduction's
1,105 steps in the same 300s — on-policy states cost marginally more per
step, consistent with them being a one-time-precomputed pool lookup vs. a
cheap `q_sample` call, though the difference is small). Final 50-step average
training loss: **0.1332** (vs. the reproduction's **0.0913** at the same
wall-clock budget — the on-policy loss curve tracks *above* the data-marginal
curve throughout training; see `logs/plots/onpolicy_training_curve.png`).
This is expected: on-policy states span a higher-variance distribution over
each `t_i` (correlated with the specific rollout that reached them) than the
data-marginal states, so the regression problem itself is harder, not a bug.

**Total on-policy run wall-clock (pool + train): 345.6s (5.76 min)** — well
inside the 20-30 min time-box; **no budget extension was used or needed to
reach this result.**

## Results — all three arms, same fixed-noise eval batch (seed 12345, n=64)

| metric | teacher (32-step) | data-marginal student (16-step, reused from repro) | on-policy student (16-step, this run) |
|---|---|---|---|
| trajectory pixel MSE vs. teacher-32 | 0.000 | 0.9277 | **1.2701** |
| trajectory classifier-L2 vs. teacher-32 | 0.000 | 0.9870 | 0.9900 |
| trajectory classifier-KL vs. teacher-32 | 0.000 | 2.9359 | 2.9846 |
| own quality: mean top-1 confidence | 0.8390 | 0.8624 | 0.8734 |
| own quality: mean entropy (nats) | 0.4467 | 0.5416 | 0.5352 |
| **population class-entropy (nats, max 2.303)** | **1.6034** | **0.0000** | **0.0000** |
| **fraction of samples in majority class** | **0.3125** | **1.0000** | **1.0000** |
| predicted-class histogram (n=64) | 7 classes represented | **all 64 → class "8"** | **all 64 → class "8"** |
| wall-clock ms/sample (sampling only) | 9.08 | 3.66 | 4.10 |

Full JSON: `logs/results.json`. Sample grid (teacher / data-marginal /
on-policy, same starting noise): `logs/plots/sample_grid_comparison.png`.
Training curves overlay: `logs/plots/onpolicy_training_curve.png`.

## Honest diagnosis — did on-policy training fix the collapse?

**No. The collapse is unchanged, and on one axis (trajectory fidelity) it is
measurably worse.**

- **Population diversity, the metric that actually revealed the original
  collapse, is identical between the two 16-step students**: both predict
  class "8" for **all 64/64** fixed-noise eval samples (population entropy
  0.000 nats, majority fraction 1.000, vs. the teacher's own 1.603 nats /
  0.3125). On-policy training states did not give the student any more
  reason to produce a diverse population of outputs — it converged to
  exactly the same degenerate "always predict one confident digit" solution.
- **The qualitative sample grid confirms this is not a metric artifact.**
  `sample_grid_comparison.png` shows the teacher's row with clear digit-like
  stroke structure and both student rows (data-marginal and on-policy) as
  visually indistinguishable unstructured noise/static — neither looks more
  digit-like than the other.
- **Trajectory-matching fidelity to the teacher, the primary metric this
  component is about, is slightly *worse* under on-policy training** (pixel
  MSE 1.2701 vs. 0.9277 for the data-marginal baseline — both are in the
  "essentially decorrelated samples" range given images live in roughly
  [-1,1], so this is a difference in degree of failure, not in kind, but it
  is not the improvement the hypothesis predicted).
- **The one axis that moved a little in the "healthier" direction is the
  per-sample classifier-judge confidence** (0.8734 vs. 0.8624), but this is
  exactly the metric the reproduction's RUN_LOG already flagged as
  misleading in isolation (both arms hit the same 100%-majority-class
  collapse regardless), so this small delta is not read as evidence of a
  real fix.
- **Training loss did decrease sensibly** (2.67 → 0.133 average, a similar
  shape to the data-marginal run's 2.7-ish → 0.09) — the on-policy training
  procedure runs correctly and optimizes its own objective; it simply
  reaches a different, still-collapsed optimum in the same 300s budget,
  slightly less-converged (0.133 vs 0.091) since on-policy targets are a
  harder regression problem in this budget.

**This directly contradicts the exposure-bias hypothesis as a *sufficient*
explanation for the collapse diagnosed in the reproduction.** Training the
student on states drawn from the model's own actual rollout trajectory,
rather than the data marginal, does not fix the population mode collapse —
the student still finds the same cheap "predict one classifier-favored
digit for nearly every input" shortcut, using the same wall-clock and step
budget. Per `extension/DECISION.md`'s pre-registered "no forced narrative"
framing, this outcome (rather than a clean fix) is itself the useful,
honestly-reported finding: it points *away* from a train/test-distribution-mismatch
explanation and *toward* toy-scale model capacity (a 309k-parameter shared
network trying to represent 16 different two-step composite mappings
simultaneously, on a T=50 base schedule) or the unweighted-MSE loss
(documented deviation from the paper, carried over unchanged here) as more
likely primary causes — narrowing, not answering, what the full-scale
RunPod follow-up needs to check first.

## Deviations from extension/DECISION.md and why

1. **None of substance.** The plan's four numbered design points were
   implemented as specified: same frozen teacher/schedule/round, on-policy
   replay-pool training states, identical hyperparameters between arms, same
   metrics (including the population-diversity metrics carried forward as
   first-class), same sample-grid format.
2. **Pool size (4,096) and pool_batch (512) were not pre-specified in
   DECISION.md** (which only specifies the general mechanism); chosen after
   a quick timing check (Bash smoke test, not part of this log's headline
   numbers) showing a full 32-step teacher rollout over a 512-batch takes
   ~5s on this CPU sandbox, so a pool of this size fits comfortably inside
   the 20-30 min time-box alongside a full 300s training run. This is a
   compute-budgeting choice, not a result-shopping one — it was fixed before
   looking at any collapse/no-collapse outcome.
3. **Time-box**: total on-policy run (pool + train) was 345.6s (5.76 min),
   well under the 20-30 min cap — the run did not need the extra budget, and
   per the instructions the budget was not extended in search of a different
   outcome once the result was in.

## What's next

This extension answers the question it set out to test: on-policy training
states alone do not resolve the toy-scale progressive-distillation collapse
this component reproduced. A further, out-of-scope follow-up (not run here,
per the single-round time-box) would isolate model capacity vs. loss
weighting as the next candidate cause -- e.g. re-running the same on-policy
scheme with a larger TinyUNet (more base channels) or with the paper's
truncated-SNR loss weighting restored, to see whether either moves the
population-diversity numbers off zero. See `../report/REPORT.md` for the
full component write-up and the RunPod scaffold for a full-scale test where
capacity is no longer a toy-scale bottleneck.
