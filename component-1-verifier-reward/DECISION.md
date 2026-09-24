# Component 1 — paper selection decision

## Candidates considered (scout pass, 2026-09-22)

1. **DDPO / DPOK** — "Training Diffusion Models with Reinforcement Learning" (Black, Janner, Du,
   Kostrikov, Levine; ICLR 2024; arXiv 2305.13301) and "DPOK: Reinforcement Learning for
   Fine-tuning Text-to-Image Diffusion Models" (Fan et al.; NeurIPS 2023; arXiv 2305.16381).
   Public, maintained, single-GPU/CPU-friendly code (`github.com/kvablack/ddpo-pytorch`).
2. **GvU** — "Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding
   for Unified Multimodal Models" (CVPR 2026; arXiv 2603.06043). The most literal fit for the
   proposal's "understanding as a generation reward" question, but no public code and built on
   a multi-billion-parameter unified backbone (X-Omni) — not reproducible at toy scale.
3. **Ask, Solve, Generate** — self-consistency rewards for unified LMMs (arXiv 2606.27376,
   code available). Same problem as (2): every base model it runs on (BAGEL/VARGPT/BLIP3o) is
   multi-billion-parameter; no small config path.

Full scout report is preserved in the session log; not duplicated here.

## Decision: anchor on DDPO/DPOK

**Reasoning:** GvU and Ask-Solve-Generate are thematically closer to the proposal's exact
language ("understanding as a training signal for generation," internal verifier), but neither
has a code path to an honest small-scale reproduction in a CPU-only sandbox — attempting either
directly would produce something "inspired by" the paper rather than a faithful reproduction of
its core claim, which violates the program's honesty bar.

DDPO/DPOK's core claim — that an arbitrary, non-differentiable scalar reward/verifier can be
turned into a training signal for a diffusion model via policy-gradient RL over the denoising
chain treated as an MDP — is mechanistically the direct ancestor of the "verifier reward" idea
and is reproducible faithfully at toy scale with public reference code to check against.

**The extension closes the gap to the actual proposal thesis.** Once the DDPO baseline (fixed,
external verifier — a frozen small classifier) is reproduced and verified, the extension
swaps in an *internal* verifier that shares representations with the generator itself (features
derived from/trained jointly with the diffusion model, not a frozen external network) — this is
a small-scale analogue of GvU's teacher-student framing, and it directly tests the proposal's
actual question ("can understanding generalize as a training signal for generation?") rather
than just reproducing DDPO as-is. A second natural ablation this setup supports: **dense
per-step reward vs. terminal-only reward** (the original DDPO reward is technically terminal,
applied at the final denoised image, even though credit is assigned at every step) — directly
testing the proposal's "dense RL-style reward" language.

## Toy-scale design (CPU-only sandbox, no GPU)

- **Dataset**: MNIST, downsampled to 16×16, grayscale.
- **Base generator**: small UNet DDPM (~150-400k params), trained from scratch, T=50 diffusion
  steps (vs. 1000 in the paper) to keep both pretraining and the RL loop's per-sample chain
  cost tractable on 4 CPU cores.
- **Verifier / reward (baseline)**: a small frozen CNN classifier trained on MNIST
  (~50-150k params), reward = classifier confidence that a sample is a target digit class.
- **RL fine-tuning**: DDPO-style policy gradient (REINFORCE with per-step log-probs of the
  Gaussian reverse transitions, reward broadcast to all steps, standardized/whitened reward as
  advantage), reproducing Algorithm 1 of the DDPO paper at this toy scale, with a KL-to-base
  penalty available as a DPOK-style option if reward hacking appears.
- **Metrics logged**: mean reward on frozen eval batch before/after RL fine-tuning, reward
  curve over RL iterations, KL-to-base-model estimate, sample grids, and any reward-hacking
  qualitative failure mode (matching DDPO §5's own honest reporting of this failure mode at
  full scale).
- **Time-box**: base DDPM pretraining and DDPO fine-tuning each capped at roughly 45-60 minutes
  of wall-clock CPU time in this sandbox. If reward does not move in that budget, that is
  itself a logged (negative) result with diagnosis, not a reason to silently scale up unboxed.

## What ships to RunPod later

A `Dockerfile` + `requirements.txt` + `runpod_launch.sh` under `repro/` (and later
`extension/`) that run the identical code at full paper scale (Stable-Diffusion-sized UNet,
CIFAR/LAION-derived data, T=1000, ImageReward/aesthetic reward) on an A100 pod — same
mechanism, just resourced up. Not executed in this session.
