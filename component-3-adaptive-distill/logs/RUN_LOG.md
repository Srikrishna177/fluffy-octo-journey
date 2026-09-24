# Progressive Distillation toy-scale reproduction — run log

Sandbox: 4 CPU cores, 15GB RAM, no GPU, PyTorch 2.14.0+cu130 (CPU-runnable
build already installed, confirmed before starting — no reinstall needed).
All numbers in this file come from runs actually executed in this sandbox
and logged to `logs/results.json` / `logs/curves/distill_train.jsonl`; none
are fabricated or projected.

## Setup

- **Teacher**: Component 1's frozen TinyUNet DDPM, `ddpm_base_teacher.pt`
  (copy of `component-1-verifier-reward/repro/checkpoints/ddpm_base.pt`,
  loaded read-only, never modified) — 308,641 params, MNIST-16x16,
  epsilon-prediction, T=50 linear beta schedule (1e-4 → 0.02), originally
  trained with ancestral DDPM sampling.
- **New code this component adds**: `repro/src/diffusion.py` (deterministic
  DDIM sampler, eta=0, plus schedule-halving utilities on the same T=50
  grid) and `repro/src/distill.py` (the progressive-distillation cascade).
  `unet.py`, `classifier.py`, `data.py`, `utils.py` are reused verbatim from
  Component 1's `repro/src/` (see file headers).
- **32-step DDIM schedule**: `make_ddim_schedule(T=50, n_steps=32)` — an
  evenly-spaced descending subsequence of `{0,...,49}`:
  `[49,47,46,44,43,41,40,38,36,35,33,32,30,28,27,25,24,22,21,19,17,16,14,13,
  11,9,8,6,5,3,2,0]`. Halving (`schedule[0::2]`) gives 16 → 8 → 4 → 2 → 1
  step schedules that are *exact subsequences* of the 32-step schedule, so
  "two old steps = one new step" is exact at every round, not approximate.
- **Distillation objective (paper's Algorithm 2, x0-parameterization)**: for
  a training pair `(t_i, t_mid, t_end)` from the current (pre-halving)
  schedule, sample `x0 ~ p_data` (real MNIST-16x16), `eps ~ N(0,I)`, form
  `z_ti = alpha_ti*x0 + sigma_ti*eps` (same marginal used for ordinary DDPM
  training — this matches the paper's own training procedure, which samples
  from the data marginal rather than rolling the teacher out from pure
  noise). Run the **teacher's** two actual DDIM steps `z_ti → z_mid → z_end`
  (no gradient). Analytically invert for the target the **student** should
  predict: `x_target = (z_end - ratio*z_ti) / (sigma_end - ratio*sigma_ti)`,
  `ratio = alpha_end/alpha_ti` (this is exactly solving "what x0 would a
  single DDIM step from `z_ti` need to land on `z_end`"). Student loss:
  `MSE(x0_student, x_target)` where `x0_student` is the student's own
  epsilon-prediction network's implied x0 estimate at `(z_ti, t_i)`. This
  is an **unweighted** version of the paper's loss (the paper's truncated-SNR
  weighting is skipped for toy-scale simplicity — documented deviation
  below). At the start of each round the student is initialized as a deep
  copy of the current teacher's weights, per spec.

## Cascade run

Command:
```
python3 distill.py --n_train 10000 --batch_size 64 --lr 2e-4 \
  --round_budget_sec 300 --round_max_steps 4000 --total_budget_sec 3300 \
  --n_eval 64 --eval_seed 12345
```
Real MNIST loaded successfully (`used_real_mnist=true`, 10,000 train images,
same loader as Component 1). Each round capped at 300s wall-clock (5 min,
the lower end of DECISION.md's 5-8 min-per-round range, matching the
program's house convention of using the lower bound to leave in-session
headroom) or 4000 steps, whichever comes first — every round in this run hit
the **time** cap, not the step cap. Total cascade wall-clock: **1500.7s
(25.0 min)**, well inside the ~45-60 min budget.

| round | transition | steps completed | wall-clock | final 50-step-avg train loss |
|---|---|---|---|---|
| 1 | 32→16 | 1105 | 300.1s | 0.091 |
| 2 | 16→8  | 1676 | 300.0s | 0.427 |
| 3 | 8→4   | 2425 | 300.0s | 0.649 |
| 4 | 4→2   | 3262 | 300.0s | 0.265 |
| 5 | 2→1   | 3951 | 300.0s | 0.111 |

The cascade **ran all the way to NFE=1** (5/5 halving rounds) — my
pre-registered collapse-stop rule (trajectory-MSE > 3x the previous round,
or own-quality classifier confidence < 0.15) never triggered. See "Honest
diagnosis" below for why that rule was the wrong instrument for the failure
mode this run actually hit, and why NFE=1 completing is not the same as the
distillation having worked.

## Results table (every NFE reached; `logs/results.json`, `logs/curves/distill_train.jsonl`)

Fixed batch of 64, fixed starting-noise seed 12345, identical across every
arm (fair trajectory comparison). Teacher = frozen `ddpm_base_teacher.pt` on
the 32-step DDIM schedule (no training).

| NFE | pixel MSE vs. teacher-32 | classifier KL(teacher‖arm) | own top-1 conf | own entropy | pred-class pop. entropy (nats, max 2.303) | majority-class frac | wall-clock ms/sample |
|---|---|---|---|---|---|---|---|
| 32 (teacher) | 0.000 | 0.000 | 0.839 | 0.447 | 1.603 | 0.313 | 6.95 |
| 16 | 0.928 | 2.936 | 0.862 | 0.542 | **0.000** | **1.000** | 3.59 |
| 8  | 0.829 | 3.811 | 0.852 | 0.431 | 1.091 | 0.656 | 1.89 |
| 4  | 1.058 | 4.582 | 0.782 | 0.578 | 0.724 | 0.766 | 0.86 |
| 2  | 0.685 | 2.499 | 0.776 | 0.619 | 1.103 | 0.656 | 0.52 |
| 1  | 0.823 | 3.276 | 0.846 | 0.449 | 0.274 | 0.922 | 0.20 |

Plots: `logs/plots/distill_training_curves.png` (per-round loss curves),
`logs/plots/metrics_vs_nfe.png` (all 6 panels above vs. NFE),
`logs/plots/sample_grid_by_nfe.png` (teacher vs. every arm, same starting
noise, 8 of the 64 eval samples per row).

## Honest diagnosis — this is the headline finding

**Trajectory-matching fidelity to the teacher's 32-step output collapses
already at the very first halving round (32→16), not gradually and not
only at the 1-2 step extreme anticipated in DECISION.md.** Pixel MSE jumps
from 0 (identical arm) to 0.93 at 16 steps and never recovers — it stays in
the same 0.68-1.06 range through every subsequent round. Given the images
live in roughly `[-1,1]`, an MSE near 0.9-1.0 is close to what two
*decorrelated* samples would show; the 16-step arm's output is essentially
unrelated to the specific teacher trajectory it was distilled from.

**The classifier-judge "own quality" confidence metric (mean top-1
confidence 0.78-0.86 across every arm, comparable to the teacher's own
0.839) is misleading here, and only inspecting it would have hidden the
real failure.** The sample grid (`sample_grid_by_nfe.png`) makes this
directly visible: the teacher row shows digit-like blob/stroke shapes; every
distilled-student row (16 down to 1 steps) looks like unstructured
high-frequency noise/static, not digits. Cross-checking against the
classifier's **predicted-class histogram** confirms this is genuine
**mode collapse**, not a metric artifact: the 16-step arm predicts class "8"
for **all 64/64** eval samples (population entropy of the predicted-class
histogram: 0.00 nats, vs. the teacher's own 1.60 nats and a max of 2.30 for
a uniform 10-class spread); the 1-step arm predicts class "8" for 59/64
(92%). The per-sample "own quality" confidence/entropy metrics specified in
DECISION.md do not see this because they average over independent samples
without checking whether those samples are all the *same* output — this is
a real limitation of the pre-registered metric set, and
`population_class_entropy_nats` / `frac_samples_in_majority_class` were
added post-hoc to `results.json` specifically to make the collapse visible
quantitatively (not to replace or override the originally-specified metrics,
which are also reported in full above).

**Diagnostic ablation confirms the collapse is caused by the distillation
training step itself, not by DDIM step-count reduction in general.**
Directly subsampling the *frozen, never-retrained* teacher network to 16 or
even 8-step DDIM (no distillation at all) produces samples nearly identical
to its own 32-step baseline (15/16 identical argmax predictions in an
independent 16-image check), confirming the base model's ODE trajectory is
smooth enough that step-count reduction alone is not the problem.

**Diagnostic ablation on learning rate and collapse-onset speed** (not part
of the main cascade; run separately and not folded into the reported
results table above): retraining round 1 (32→16) from scratch with a 10x
lower learning rate (2e-5 vs. the main run's 2e-4) still collapses to
100% class "8" in 440 steps (120s budget). A step-by-step checkpoint sweep
at the main run's lr=2e-4 shows the collapse is fast, not gradual: the
16-step student already predicts class "8" for 64/64 fixed-noise samples
after just **20 gradient steps** (out of the 1105 the full round used),
while its own regression training loss is still around 0.79 (well above its
eventual converged value of 0.091) — i.e. the *sampling*-time mode collapse
sets in almost immediately, long before the *training*-time regression loss
has meaningfully converged, and is insensitive to a 10x learning-rate
change. This rules out "learning rate too high / training ran too long" as
the explanation.

**Likely mechanism (not conclusively isolated further given the time-box):**
the population-marginal training scheme specified by the paper's own
Algorithm 2 — sampling `z_ti` from `alpha_ti*x0 + sigma_ti*eps` for real
MNIST `x0` and fresh noise, rather than from the model's own partial
sampling rollout — supervises the *shared* 309k-parameter network at 16
different, only data-marginal-realistic states per round. Gradient descent
on the combined multi-condition regression loss appears to find a cheap
shared optimum (predict something the frozen classifier confidently reads
as a single fixed digit, for close to every input) that lowers the
population-marginal MSE loss without the network actually learning the
per-condition two-step composite mapping needed for a coherent
noise-to-image *sequential* rollout starting from pure noise (a state the
training distribution never explicitly constructs). This is a form of
exposure bias / train-test mismatch specific to this training scheme at
this small model capacity and short base schedule (T=50) — plausibly less
severe at the paper's actual scale (much larger UNets, thousands of base
steps, weighted loss, EMA), which is exactly why DECISION.md correctly
scoped this as a toy CPU reproduction rather than a claim that Progressive
Distillation itself doesn't work.

**Bottom line**: the progressive-halving *mechanism* (schedule construction,
teacher-freeze/copy-init cascading, DDIM sampler, the paper's own
Algorithm-2 regression target) runs end-to-end without a single crash or
NaN, and training loss decreases sensibly within every round. But the
core claim under test — that the resulting few-step student approximates
the many-step teacher's *sampling trajectory* — **fails already at the
first halving (32→16 steps)** in this toy setup: the student's own-sample
"quality" looks superficially fine only because it collapsed onto a narrow
set of classifier-favored outputs, which the qualitative sample grid and
the added population-diversity metrics both show plainly. This is reported
as the honest, negative headline result of this reproduction, consistent
with the program's honesty bar — not tuned away, and not re-run with new
hyperparameters after seeing the numbers except as clearly-labeled
diagnostic ablations (above) aimed at understanding, not improving, the
already-reported main-run numbers.

## Deviations from DECISION.md and why

1. **Collapse-stop rule never triggered, despite what turned out to be the
   real failure.** DECISION.md's proposed thresholds (trajectory-MSE > 3x
   the previous round, or own-quality confidence collapsing toward random)
   were implemented as specified (`--collapse_ratio 3.0`,
   `--collapse_conf_floor 0.15`) but neither caught this run's actual
   failure mode (an *immediate*, *stable-magnitude* trajectory-MSE jump at
   round 1 that never worsens further, paired with *misleadingly high*
   per-sample confidence due to population mode collapse rather than random
   guessing). The cascade was therefore allowed to run to completion (which
   is informative in its own right — it shows the degradation is front-loaded
   at 32→16 rather than progressively worse at low NFE) rather than
   stopping early. This is flagged here rather than silently declaring
   "success" because the cascade reached NFE=1.
2. **Unweighted MSE loss** instead of the paper's truncated-SNR weighting —
   simplicity at toy scale, documented in Setup above. Given the collapse
   traces to *population mode collapse*, not to a training-loss-weighting
   issue (confirmed via the learning-rate/step-count ablations), this is
   unlikely to be the primary cause, but is not ruled out as a contributing
   factor and is the first thing to revisit in a follow-up.
3. **`n_train=10000`, `batch_size=64`, `lr=2e-4`** reused from Component 1's
   DDPM training defaults rather than re-tuned — kept for consistency with
   the frozen teacher's own training regime.
4. **Two small labeled diagnostic ablations** (learning rate, collapse-onset
   speed) were run after seeing the main cascade's results, specifically to
   diagnose *why* the collapse happened, not to search for hyperparameters
   that avoid it. They are reported in full above (including the negative
   finding that a 10x lower LR does not fix the collapse) and are kept
   clearly separate from the main results table, per the program's honesty
   bar against post-hoc result-shopping.
5. **Metric addition**: `population_class_entropy_nats` and
   `frac_samples_in_majority_class` were added to `results.json` after the
   main run (computed from the already-saved fixed-noise sample batches, no
   retraining), because the originally-specified per-sample metrics failed
   to surface a mode-collapse failure that was clearly visible in the
   qualitative sample grid. This is disclosed as a real gap in the
   pre-registered metric set, not hidden.

## What's next

Given the honest finding above, the natural extension question sharpens
further: does the DECISION.md-proposed Consistency-Model-style single-step
objective (self-consistency along the ODE trajectory, rather than
population-marginal 2-step regression) avoid this specific population
mode-collapse failure mode, or hit the same toy-capacity ceiling from a
different angle? That is out of scope for this reproduction pass and
belongs to the extension phase.
