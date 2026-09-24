# Component 3 — paper selection decision

## Candidates considered (scout pass, 2026-09-24)

1. **Progressive Distillation for Fast Sampling of Diffusion Models** (Salimans & Ho, ICLR
   2022; arXiv 2202.00512). A trained N-step deterministic (DDIM) sampler is distilled into a
   student matching the teacher's *2* steps with *1* student step; iterating halves the
   required steps each round. A literal, clean instance of "many small iterative-refinement
   passes -> few/one pass, comparable quality, measured compute." Multiple public toy
   reimplementations; fully CPU-feasible at toy scale.
2. **PonderNet** (Banino, Firoiu, Blundell, 2021; arXiv 2107.05407). Learned per-input halting
   depth, trained to balance accuracy against expected compute; matches/beats fixed-depth ACT
   baselines using far fewer average steps on parity/bAbI. Cleanest ACT-branch fit, cheapest to
   reproduce (tiny RNN, minutes on CPU), but a looser fit to the proposal's "distill a multi-turn
   refinement loop" framing (it's adaptive depth vs. a fixed-depth baseline, not literally
   distilling an explicit critique-and-regenerate loop).
3. **Distilling System 2 into System 1** (Yu, Xu, Weston, Kulikov; Meta, 2024; arXiv
   2407.06023). Multi-step LLM self-critique/rewrite techniques generate synthetic
   (input, final-answer) pairs; a single-pass model is fine-tuned on those pairs. The most
   conceptually direct match to the proposal's literal LLM-refinement framing, and it honestly
   reports its own negative result (complex multi-step math/CoT doesn't distill well). No
   public code; would need a full prompted-multi-turn-pipeline reimplementation at toy scale —
   most engineering-heavy of the three.

Full scout report preserved in the session log; not duplicated here.

## Decision: anchor on Progressive Distillation, extend toward a Consistency-Model-style single-step objective

**Reproduction anchor: Progressive Distillation** (candidate 1). Best combination of a
literal fit to the research question, public reference code, and a toy-scale path that's
fully CPU-feasible with an unambiguous quantitative artifact (a quality-vs-NFE curve) rather
than a fragile prompted-LLM pipeline.

**Efficient reuse of existing infrastructure**: rather than training a new teacher from
scratch, this component reuses **Component 1's already-trained, frozen TinyUNet DDPM**
(`component-1-verifier-reward/repro/checkpoints/ddpm_base.pt`, ~309k params, MNIST-16x16,
trained with T=50 ancestral DDPM) as the multi-step teacher. This is exactly the "many small
refinement forward passes" baseline this component is about, and reusing it avoids redundant
pretraining, mirroring how Component 1's own extension reused its own reproduction's
checkpoints.

**Extension: distill toward a Consistency-Model-style objective** (Song, Dhariwal, Sutskever,
ICML 2023) — instead of iterative step-halving, train a single-step generator directly via a
self-consistency objective (mapping any point on the same ODE trajectory to the same clean
sample) and compare its one-step quality against the progressively-distilled few-step
cascade at matched NFE. This mirrors the established pattern: Component 1 anchored on
DDPO then extended toward an internal-verifier mechanism; Component 2 anchored on MaskGIT
then extended toward a learned-order mechanism; Component 3 anchors on progressive halving
then extends toward the field's more direct single-pass answer to the same question.

## Toy-scale design (CPU-only sandbox, no GPU)

- **Teacher**: Component 1's frozen TinyUNet DDPM, switched from ancestral sampling to a
  **deterministic DDIM sampler** (needed for progressive distillation's "2 teacher steps = 1
  student step" pairing to be well-defined) at a convenient power-of-two step count subsampled
  from the underlying T=50 schedule (32 steps).
- **Progressive halving cascade**: 32 -> 16 -> 8 -> 4 -> 2 -> 1 steps. Each round initializes
  the student from the previous round's weights (or the teacher, for the first round) and
  trains it to match two consecutive steps of the current teacher with one student step,
  using the standard progressive-distillation regression loss. If a round's quality collapses
  badly (see metric below), stop the cascade there and report the best step count reached
  rather than pushing further into a regime that isn't working — that is itself a valid,
  logged finding about where this toy setup's distillation breaks down.
- **Primary metric — trajectory-matching fidelity**: for a shared batch of fixed random
  starting noise, compare the *teacher's* full-schedule (32-step) output against each
  *distilled student's* output from the *same* starting noise, via pixel MSE (and, as a
  secondary check, the frozen MNIST classifier's confidence-vector distance) — this is the
  standard progressive-distillation evaluation (does the student approximate the teacher's
  own sampling trajectory), not a comparison to real MNIST ground truth.
- **Secondary metric — absolute quality anchor**: classifier-judge score (mean top-1
  confidence, entropy) on each arm's unconditional samples, reusing Component 1's frozen
  classifier, to sanity-check that quality is reasonable in absolute terms, not just close to
  a possibly-poor teacher.
- **Compute/latency axis**: number of function evaluations (NFE) per sample (32, 16, 8, 4, 2,
  1) and measured wall-clock sampling time per arm — the actual "latency budget" this
  component's proposal names.
- **Time-box**: each halving round capped at ~5-8 minutes wall-clock (should converge fast
  since it's a regression against a fixed teacher, not from-scratch training); total cascade
  budget ~45-60 min. If time runs out partway through the cascade, report however many rounds
  completed rather than rushing the rest.

## What ships to RunPod later

`Dockerfile` / `requirements.txt` / `runpod_launch.sh` under `repro/` running the identical
mechanism at full scale (a real Stable-Diffusion-sized UNet, the paper's original CIFAR/
ImageNet setting, starting from a much larger step count). Not executed in this session.
