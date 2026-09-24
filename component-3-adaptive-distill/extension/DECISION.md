# Component 3 — extension design

## Recap of reproduction result (revised extension plan — see note below)

The reproduction's progressive-halving cascade ran end-to-end without crashing, but the core
claim it was testing — that a distilled few-step student approximates the many-step teacher's
sampling trajectory — failed immediately at the *first* halving (32->16 steps): the student's
samples collapsed to unstructured noise, masked by a misleadingly-stable classifier-confidence
score caused by population mode collapse (e.g. the 16-step arm predicted class "8" for 64/64
fixed-noise samples). Diagnostics ruled out learning rate and confirmed the frozen teacher
itself samples fine at reduced step counts with no retraining — the collapse is specific to
the distillation training step. Full detail: `../logs/RUN_LOG.md`.

**Note on scope change**: `DECISION.md`'s original extension plan (distill toward a
consistency-model-style single-step objective) is deferred. Running that on top of a base
method that already collapses at the first halving would confound two unknowns (does the new
objective fail, or does the same underlying issue sink it too?) and would not be a meaningful
test of anything. Instead, this extension directly tests the reproduction's own diagnosed
mechanism — the most scientifically prior, best-motivated next question raised by the actual
data, per the program's time-boxing rule to redirect rather than continue by default down a
path that isn't teaching much.

## The question this extension tests

The reproduction's RUN_LOG.md hypothesizes the collapse is a form of **exposure bias**: the
paper's own Algorithm 2 trains the student on noisy states `z_t` sampled from the *data*
marginal (`alpha_t*x0 + sigma_t*eps` for real MNIST `x0` and fresh noise), never from the
model's own actual sampling rollout starting at pure noise — so gradient descent can find a
cheap shortcut (predict one classifier-favored digit for nearly every input) that lowers the
population-averaged training loss without learning the real sequential composite mapping.

**Does training the student on states drawn from the teacher's own rollout trajectory (starting
from pure noise, matching what the student will actually see at sampling time) rather than from
the data marginal fix the collapse?** This is a direct, testable version of the diagnosed
exposure-bias hypothesis, not a cosmetic variation.

## Design

1. **Keep everything else fixed**: same frozen teacher (Component 1's TinyUNet DDPM), same
   32-step DDIM schedule, same first halving round (32->16) — the round where collapse was
   sharpest and earliest, so this is the cleanest single round to test the fix on before
   deciding whether to re-run the full cascade.
2. **On-policy training targets**: instead of sampling `z_t` from the data marginal, generate
   training states by rolling the *teacher* out from pure noise using its actual DDIM sampling
   procedure, and take the (state, timestep) pairs actually visited along that rollout as the
   training inputs (with the same 2-step-teacher / 1-step-student regression target as before,
   computed from that on-policy state rather than a data-marginal one).
3. **Comparison**: train two 16-step students under identical hyperparameters (same
   architecture, init, lr, step budget) — one with the reproduction's original data-marginal
   scheme (already have these numbers from the reproduction, reusable), one with the new
   on-policy scheme — and evaluate both with the exact same metrics as the reproduction
   (trajectory-matching MSE/classifier-KL vs. teacher, own classifier-judge score, AND the
   population-diversity metrics added during the reproduction's diagnosis, since those are
   what actually revealed the failure mode — carrying them forward as first-class metrics
   here, not an afterthought).
4. **Sample grids**: same shared-starting-noise visual comparison format as the reproduction,
   teacher vs. data-marginal-student vs. on-policy-student, so a collapse (or its absence) is
   visible directly, not just inferable from scalar metrics that were shown to be misleading
   on their own.

## Hypothesis and honest expectation

No forced narrative:
- If on-policy training fixes the collapse (diverse, teacher-like samples at 16 steps) →
  supports the exposure-bias diagnosis and suggests a concrete, defensible fix path for the
  full-scale RunPod follow-up.
- If on-policy training still collapses → the exposure-bias hypothesis was incomplete or
  wrong, and the real cause is more likely toy-scale model capacity (309k params, T=50 base
  schedule) rather than the training-state distribution — also a real, useful finding that
  narrows down what full-scale validation would need to check.
Whichever happens gets reported plainly.

## Time-box

On-policy rollout generation adds real cost (each training example now requires running the
teacher's own multi-step DDIM sampler, not just a single data-marginal noise draw), so this is
capped tighter than the reproduction: ~20-30 min total, one halving round only (32->16), not a
full re-cascade. If on-policy training doesn't clearly resolve the collapse within budget,
that is the answer — report it, do not extend the budget hoping for a different outcome.
