# Component 3 — Adaptive-Compute Distillation

*Reproduction + extension report. ~10-minute read. All numbers below are
copied or directly computed from `component-3-adaptive-distill/logs/
results.json` and `component-3-adaptive-distill/extension/logs/results.json`
— cross-checked against those files before writing, not from memory.*

## 1. The paper reproduced

**Progressive Distillation for Fast Sampling of Diffusion Models** (Salimans
& Ho, ICLR 2022). Given a trained deterministic (DDIM) N-step sampler, train
a student to match the teacher's output after *two* consecutive teacher
steps using only *one* student step. Repeating the procedure — student
becomes the new teacher — halves the required sampling steps each round:
32 → 16 → 8 → 4 → 2 → 1. This is a direct instance of this program's
Component 3 question: can many small iterative-refinement passes be
distilled into far fewer passes at comparable quality, under a measured
compute/latency budget (number of function evaluations, NFE).

**Toy-scale reproduction design** (`component-3-adaptive-distill/DECISION.md`):
rather than training a new teacher, this component reuses **Component 1's
already-trained, frozen TinyUNet DDPM** (308,641 params, MNIST-16x16,
originally trained with T=50 ancestral DDPM) as the teacher, switched to a
deterministic 32-step DDIM sampler. A full progressive-halving cascade
(32→16→8→4→2→1) was implemented per the paper's own Algorithm 2 regression
target, with an unweighted MSE loss in place of the paper's truncated-SNR
weighting (a documented, toy-scale simplification).

## 2. Reproduction's real finding: collapse at the very first halving

The cascade ran end-to-end without a single crash or NaN, all the way to
NFE=1, in **25.0 minutes wall-clock** (well under the 45–60 min budget).
But the core claim under test — that a distilled few-step student
approximates the many-step teacher's *sampling trajectory* — failed
immediately, at the **first** halving (32→16 steps), not gradually toward
1–2 steps as the design anticipated:

| NFE | pixel MSE vs. teacher-32 | classifier-KL vs. teacher-32 | own top-1 conf. | population class-entropy (nats, max 2.30) | fraction in majority class |
|---|---|---|---|---|---|
| 32 (teacher) | 0.000 | 0.000 | 0.839 | **1.603** | **0.31** |
| 16 | 0.928 | 2.936 | 0.862 | **0.000** | **1.00** |
| 8 | 0.829 | 3.811 | 0.852 | 1.091 | 0.66 |
| 4 | 1.058 | 4.582 | 0.782 | 0.724 | 0.77 |
| 2 | 0.685 | 2.499 | 0.776 | 1.103 | 0.66 |
| 1 | 0.823 | 3.276 | 0.846 | 0.274 | 0.92 |

The per-sample classifier-confidence column looks fine throughout (0.78–0.86,
comparable to the teacher's 0.839) — this is the misleading part. The
population-diversity columns (added during diagnosis specifically because
the confidence metric was hiding the failure) tell the real story: the
16-step student predicts the **same class for all 64/64** fixed-noise eval
samples, versus the teacher's own healthy spread across 7 classes. This was
confirmed **visually**: the teacher's samples show clear digit-like stroke
structure; every distilled arm's samples are unstructured noise.

**Diagnostics ruled out the easy explanations.** The frozen, *never-retrained*
teacher itself samples fine when simply run at 8 or 16 DDIM steps directly
(no distillation) — so step-count reduction alone is not the problem. A
10x-lower learning rate still collapses to the same degenerate solution in
440 steps. A step-by-step checkpoint sweep showed the collapse sets in
within **~20 gradient steps**, long before the training-time regression loss
has meaningfully converged (0.79 → eventual 0.091) — the *sampling-time*
mode collapse and the *training-time* loss are decoupled. Full detail:
`component-3-adaptive-distill/logs/RUN_LOG.md`.

**Working hypothesis at the end of the reproduction**: exposure bias. The
paper's own Algorithm 2 trains on noisy states drawn from the *data*
marginal (real MNIST images + fresh noise), never from the model's own
actual sampling rollout starting at pure noise. Gradient descent on this
population-marginal objective can find a cheap shared shortcut — predict one
classifier-favored digit for nearly every input — that lowers the averaged
training loss without the network learning the real per-condition sequential
mapping it needs at sampling time.

## 3. Extension's question and design

**Does training on states drawn from the teacher's own rollout trajectory
(on-policy), rather than the data marginal, fix the collapse?** A direct,
testable version of the exposure-bias hypothesis — not a cosmetic variation,
and not the originally-planned consistency-model extension (deferred:
running a new objective on top of a method that already collapses at the
first halving would confound two unknowns at once and teach little).

Design (`component-3-adaptive-distill/extension/DECISION.md`): freeze the
same teacher and 32-step schedule; generate training states by rolling the
teacher out from pure noise on its own DDIM schedule (a 4,096-entry replay
pool of on-policy `(state, timestep)` pairs, 44.3s to generate); train a
16-step student on these states with the *identical* regression target,
architecture, init, learning rate, and step budget as the reproduction's
original round 1, so the only variable that changes is where `z_t` comes
from.

## 4. Extension's real result: the fix did not work

| metric | teacher (32-step) | data-marginal student (repro, 16-step) | on-policy student (extension, 16-step) |
|---|---|---|---|
| trajectory pixel MSE vs. teacher-32 | 0.000 | 0.928 | **1.270** (worse) |
| trajectory classifier-KL vs. teacher-32 | 0.000 | 2.936 | 2.985 |
| own top-1 confidence | 0.839 | 0.862 | 0.873 |
| population class-entropy (nats) | **1.603** | **0.000** | **0.000** |
| fraction in majority class | **0.31** | **1.00** | **1.00** |
| predicted-class histogram (n=64) | 7 classes | all 64 → class "8" | all 64 → class "8" |

Training ran correctly (loss decreased 2.67 → 0.133, a similar shape to the
reproduction's 2.7 → 0.091) and used only 345.6s of a 20–30 min budget — no
budget extension was used or needed to reach this result. But **the
population-diversity numbers, the metric that actually revealed the original
collapse, are byte-for-byte identical between the data-marginal and
on-policy students**: both collapse to predicting a single class for every
one of 64 fixed-noise samples. On the primary trajectory-fidelity metric,
on-policy training is slightly *worse*, not better. The sample grid
(`extension/logs/plots/sample_grid_comparison.png`) confirms this
qualitatively — the on-policy student's outputs are visually
indistinguishable from the data-marginal student's noise, not closer to the
teacher's digit-like structure.

**This directly contradicts the exposure-bias hypothesis as a sufficient
explanation.** Training on-policy did not give the student any reason to
produce a diverse population of outputs; it converged to the same degenerate
shortcut using the same budget. This is reported as the honest result, not
rounded up to a partial win — the one metric that moved in a "healthier"
direction (own-sample confidence, 0.873 vs. 0.862) is exactly the metric the
reproduction already flagged as misleading in isolation, since both arms hit
the same 100%-majority-class collapse regardless.

**What this points toward instead**: the reproduction's RUN_LOG reasons that
toy-scale model capacity (a single 309k-parameter network trying to
represent 16 different two-step composite mappings simultaneously, on a
short T=50 base schedule) or the unweighted-MSE loss (a documented deviation
from the paper, carried over unchanged in this extension) are now the more
likely primary causes — narrowed, not resolved, by this extension.

## 5. What this does and doesn't show at this scale

**Does show**: (1) the progressive-halving distillation *mechanism* itself
(schedule construction, teacher-freeze/copy-init cascading, DDIM sampling,
the paper's own Algorithm-2 regression target) runs correctly end-to-end
with no implementation bugs, at every step count from 32 down to 1; (2) at
this toy scale, the resulting distilled students suffer a severe, immediate
population mode collapse that a naive per-sample confidence metric fails to
detect — a real methodological lesson about which metrics to trust when
evaluating distilled generative models; (3) a natural, well-motivated fix
attempt (on-policy training states, targeting a specific diagnosed
mechanism) does not resolve it, narrowing the likely cause toward model
capacity or loss weighting rather than train/test state-distribution
mismatch.

**Doesn't show**: anything about whether Progressive Distillation "doesn't
work" — this is a 309k-parameter model, a T=50 base schedule, 10,000-20,000
MNIST training images, and an unweighted loss, nowhere near the paper's
actual scale (much larger UNets, thousands of base diffusion steps, the
prescribed SNR-weighted loss, EMA, and real image datasets). The paper's own
reported results at that scale are not challenged by this toy finding.

**Path to full-scale validation**: `component-3-adaptive-distill/repro/`
ships a RunPod-ready scaffold (`Dockerfile`, `requirements.txt`,
`runpod_launch.sh`) for rerunning the identical mechanism with a
larger UNet, a longer base schedule, and the paper's prescribed SNR-weighted
loss on an A100 pod — directly testing whether either of the two remaining
candidate causes (capacity, loss weighting) is what's driving the toy-scale
collapse found here. Not built or executed in this CPU sandbox.

## References

- Full reproduction log: `component-3-adaptive-distill/logs/RUN_LOG.md`
- Full extension log: `component-3-adaptive-distill/extension/RUN_LOG.md`
- Reproduction numbers: `component-3-adaptive-distill/logs/results.json`
- Extension numbers: `component-3-adaptive-distill/extension/logs/results.json`
- Reproduction design: `component-3-adaptive-distill/DECISION.md`
- Extension design: `component-3-adaptive-distill/extension/DECISION.md`
