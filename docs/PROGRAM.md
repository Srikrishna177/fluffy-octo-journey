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
