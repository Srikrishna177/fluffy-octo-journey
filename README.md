# PhD Research Portfolio — Verification-in-the-Loop Unified Multimodal Generation

Research portfolio supporting a PhD application in applied AI, organized around one proposal:
**"Verification-in-the-Loop Unified Multimodal Generation with Adaptive-Compute Decoding."**

The proposal decomposes into three testable components. Each ships as a standalone artifact:
a faithful small-scale reproduction of a specific paper's core claim, plus one meaningful,
well-motivated extension that tests a real question.

| # | Component | Core question | Status |
|---|---|---|---|
| 1 | Internal verifier / dense RL-style reward | Can understanding generalize as a training signal for generation? | ✅ Done — [report](component-1-verifier-reward/report/REPORT.md) |
| 2 | Difficulty-conditioned order controller | Does learned generation order beat arbitrary-order and strict left-to-right? | 🔵 In progress (repro done, negative result) |
| 3 | Adaptive-compute distillation | Can a multi-turn refine loop be distilled into one adaptive-compute pass under a latency budget? | ⚪ Not started |

See [`docs/PROGRAM.md`](docs/PROGRAM.md) for the full directive, constraints, and working
conventions this portfolio follows across sessions.

## Layout

```
component-1-verifier-reward/   # component 1 artifact (paper repro + extension + report)
component-2-order-controller/  # component 2 artifact
component-3-adaptive-distill/  # component 3 artifact
docs/                          # cross-cutting program docs
```

Each component directory is self-contained: its own README, environment setup, experiment
logs (including negative results), and a short report.

## Compute model

This portfolio is developed inside a sandboxed coding session with no GPU and no RunPod
access. Every reproduction and ablation is first built and verified at CPU-feasible toy scale
(small models, small datasets, short runs) so that every logged number in this repo is a real,
reproducible measurement — never a projected or fabricated one. Each component ships
RunPod-ready launch scripts (Dockerfile / requirements / run script) for scaling the same code
to A100-class hardware; scaling those runs and dropping the resulting logs back in is a
follow-up step outside this session.
