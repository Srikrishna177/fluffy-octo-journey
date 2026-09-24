# Program directive

This file is the standing context for anyone (human or agent) picking up work on this
portfolio in a fresh session. It summarizes the original directive and the decisions made
so future sessions don't need to re-derive them.

## Applicant background

Applied AI engineering at Cisco (LLM deployment, agentic systems), plus independent projects:
a PPO/LangGraph adaptive RAG system, an agentic fraud-detection platform, and a Kaggle
medical-imaging competition with a real experiment log. RunPod access (A100-class GPUs) is
available to the applicant outside this coding session; moderate compute budget; limited time
alongside a full-time job.

## Proposal

**Verification-in-the-Loop Unified Multimodal Generation with Adaptive-Compute Decoding**,
three components:

1. **Internal verifier / dense RL-style reward** — an internal verifier supervises generation
   with a dense RL-style reward. Tests whether understanding can generalize as a training
   signal for generation.
2. **Difficulty-conditioned order controller** — for diffusion/multimodal generation. Tests
   whether learned generation order beats both arbitrary-order and strict left-to-right.
3. **Adaptive-compute distillation** — tests whether a multi-turn refine loop can be distilled
   into a single adaptive-compute pass under a latency budget.

## Deliverable bar (per component)

Each artifact = (a) a faithful small-scale reproduction of a specific paper's core claim, and
(b) one meaningful, well-motivated extension or ablation testing a real question (not a
cosmetic addition). Ships as: a working repo, a logged experiment trail (including negative
results), and a short written report a professor could read in 10 minutes.

Success bar: "I reproduced [paper]'s core result at small scale, then tested [specific
extension] and found [specific outcome], here's the repo and the log." Not "I ran a
tutorial." Not "I built something impressive-looking with no clear question behind it."

Depth over breadth. No generic skill-building projects unless a specific fundamentals gap
blocks one of the three components.

## Operating constraints (decided at program start, 2026-09-22)

- **Compute**: this coding session has no GPU / no RunPod credentials. All reproductions and
  extensions are built and verified at CPU-feasible toy scale first, with real logged results
  (never fabricated or projected). Each component ships RunPod-ready scripts
  (Dockerfile/requirements/launch script + exact commands) so the applicant can scale the same
  code to A100 hardware themselves and drop logs back in.
- **Sequencing**: depth-first. Fully finish one component's artifact (paper selection → repro
  → extension → report) before starting the next. Current order: 1 → 2 → 3.
- **Clarifying questions**: at most 3, only when genuinely blocking. Otherwise proceed and
  checkpoint after each major phase (paper selection, reproduction, extension design,
  experiment run) rather than asking for approval.
- **Honesty**: negative or ambiguous results are reported plainly with a clear diagnosis. No
  manufactured positive results.
- **Time-boxing**: flag and propose cutting any path burning disproportionate time/compute
  relative to what it teaches.
- **Multi-agent execution**: scout agents shortlist 2-3 candidate papers per component;
  reproduction agent(s) get the core claim running at small scale and document deviations from
  the paper and why they're defensible; extension agent(s) design and run the meaningful
  addition once reproduction is verified; a write-up pass consolidates repo + logs into the
  short report and cross-checks that claims are backed by logged results.

## Status log

- 2026-09-22: Program scaffolding created. Component 1 scouting started.
- 2026-09-22/23: Component 1 done. Anchored on DDPO/DPOK; toy MNIST-16x16 reproduction
  cleanly reproduced the core claim (reward 0.067→1.0, frac-argmax-target 9.4%→100%) and
  DDPO's own documented reward-over-optimization failure mode (diversity collapse, KL-to-base
  →53 nats). Extension tested an internal (generator-feature) verifier vs. the external
  classifier reward: mixed/ambiguous result, reported honestly — qualitative evidence
  (sample grids) favors "internal verifier is easier to hack" (collapsed to a cruder,
  less digit-like shortcut); the numeric KL trajectory looked healthier but was judged a
  likely confound (residual-gradient exhaustion, not real robustness) rather than taken at
  face value. Full report: `component-1-verifier-reward/report/REPORT.md`.
- 2026-09-24: Component 2 scouted and anchored on MaskGIT (confidence-ranked parallel
  decoding). Reproduction complete: on toy MNIST-16x16 masked-token reconstruction (75% mask,
  8-step decode, 1000-image eval, one shared trained transformer), MaskGIT's confidence-based
  reveal order did NOT beat fixed baselines — it was slightly worse than both raster and
  random order (token-recon acc 0.8447 vs 0.8499/0.8493; independent classifier-judge
  top-1-match 41.2% vs 50.2%/46.5%). Visually confirmed via sample-reconstruction grids
  (confidence-order outputs are more fragmented/incoherent). Logged honestly as a negative
  result, with a pre-registered (not post-hoc) risk in DECISION.md as the likely explanation:
  confidence miscalibration at the untrained 75%-mask extreme, compounded by small model/
  vocab capacity. Full log: `component-2-order-controller/logs/RUN_LOG.md`. Proceeding to the
  extension (learned order-policy vs. this same confidence heuristic and the fixed baselines)
  — now an even sharper test, since it asks whether learning can fix what the hand-designed
  heuristic got wrong here.
- 2026-09-24: Component 2 extension complete. Trained a small MLP policy (99-dim features:
  frozen transformer's hidden state + position + confidence + entropy) on ~3.84M
  ground-truth-labeled (position, is-correct) examples from the MNIST train split, then added
  it as a 4th "learned" reveal-order arm to the reproduction's exact 1000-image eval protocol.
  Result: learned (0.8452 token-recon acc / 40.5% classifier top-1) did not clearly beat
  confidence (0.8447 / 41.2%) — the two metrics disagree on direction, indicating a
  near-zero true gap — and both remained clearly below the raster (0.8499/50.2%) and random
  (0.8493/46.5%) fixed baselines. Supports the pre-registered hypothesis "learned doesn't
  clearly beat confidence either" (the policy's own held-out accuracy, 87.4%, was only ~2
  points above a majority-class baseline of 85.4%, suggesting limited predictable signal at
  this scale, not a flawed ranking mechanism). No train/eval leakage (policy trains on MNIST
  train split, eval uses the MNIST test split — disjoint datasets). Full log:
  `component-2-order-controller/extension/RUN_LOG.md`. Component 2 artifact complete
  end-to-end: `component-2-order-controller/report/REPORT.md`. Proceeding to component 3.
- 2026-09-24: Component 3 scouted and anchored on Progressive Distillation (Salimans & Ho),
  reusing Component 1's frozen TinyUNet DDPM as the teacher with a new deterministic DDIM
  sampler. Reproduction complete: the full 32→16→8→4→2→1-step halving cascade ran to
  completion in 25.0 min wall-clock (well under the ~45-60 min budget), but the honest
  finding is a negative one that surfaced *earlier* than DECISION.md's pre-registered risk
  (expected breakdown mainly at 1-2 steps): trajectory-matching fidelity to the teacher's
  32-step output collapsed already at the first halving round (32→16 steps; pixel MSE
  0.00→0.93, never recovering in later rounds), and the per-sample classifier-judge
  "quality" metric was actively misleading — it stayed high (0.78-0.86 confidence) only
  because the distilled students had mode-collapsed onto a narrow set of classifier-favored
  outputs (16-step arm: 64/64 samples predicted class "8", population entropy 0.00 nats vs.
  the teacher's 1.60), confirmed both visually (sample grid: distilled rows look like
  unstructured noise, not digits) and via added population-diversity metrics. Diagnostic
  ablations isolated the cause to the distillation training step itself (not DDIM step-count
  reduction: the frozen, untrained teacher samples fine at 8-32 steps) and ruled out
  learning-rate tuning as a fix (10x lower LR still collapses; collapse sets in within ~20
  gradient steps, long before training loss converges). Full log:
  `component-3-adaptive-distill/logs/RUN_LOG.md`.
- 2026-09-24: Component 3 extension complete. The reproduction's own hypothesis (exposure
  bias: the paper's Algorithm 2 trains on data-marginal noisy states, never on the model's
  actual rollout from pure noise) was tested directly by generating an on-policy replay pool
  (4,096 states visited by the teacher's own 32-step DDIM rollout) and training a 16-step
  student on it with identical hyperparameters to the reproduction's round 1. Result: the fix
  did NOT work. Population diversity is byte-for-byte identical to the data-marginal
  baseline (both collapse to predicting one class for all 64/64 eval samples; population
  entropy 0.00 nats either way), and trajectory-matching fidelity to the teacher was
  slightly *worse* under on-policy training (pixel MSE 1.27 vs. 0.93), not better. Confirmed
  visually (on-policy and data-marginal sample rows indistinguishable, both unstructured
  noise vs. teacher's clear digit structure). This rules out exposure bias as a *sufficient*
  explanation and narrows the likely cause toward toy-scale model capacity or the
  unweighted-loss deviation, for a full-scale RunPod follow-up to check. Full log:
  `component-3-adaptive-distill/extension/RUN_LOG.md`. Component 3 artifact complete
  end-to-end: `component-3-adaptive-distill/report/REPORT.md`.

**Portfolio complete: all 3 components shipped** (reproduction + extension + report each),
2026-09-24.
