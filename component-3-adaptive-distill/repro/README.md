# Component 3 — Progressive Distillation, toy reproduction

Reproduction of the core mechanism in *Progressive Distillation for Fast
Sampling of Diffusion Models* (Salimans & Ho, arXiv 2202.00512), at CPU-only
toy scale, reusing Component 1's frozen TinyUNet DDPM as the teacher.

See `../DECISION.md` for the design spec and `../logs/RUN_LOG.md` for the
full narrated run (exact commands, hyperparameters, wall-clock, honest
diagnosis of where and why the cascade's trajectory-fidelity broke down).

## Layout

- `src/unet.py`, `src/classifier.py`, `src/data.py`, `src/utils.py` — reused
  verbatim from `component-1-verifier-reward/repro/src/` (frozen TinyUNet
  architecture, MNIST-16x16 loader, frozen classifier, shared utilities).
- `src/diffusion.py` — new: deterministic DDIM sampler + progressive-halving
  schedule utilities on top of the same T=50 linear noise schedule Component
  1 trained the teacher with.
- `src/distill.py` — the progressive-distillation cascade: trains
  32→16→8→4→2→1-step students, evaluates every arm, writes
  `logs/results.json` and `logs/curves/distill_train.jsonl`.
- `src/plots.py` — renders `logs/plots/*.png` from the logged JSON/JSONL/
  checkpoint artifacts (read-only, no training).
- `checkpoints/` — `ddpm_base_teacher.pt` / `classifier.pt` (copied
  read-only from Component 1), plus `student_{16,8,4,2,1}step.pt` produced by
  this run, and `eval_samples_by_nfe.pt` (the fixed-noise sample batch used
  for every arm's plots/metrics).

## Rerun

```bash
cd component-3-adaptive-distill/repro/src
python3 distill.py \
  --n_train 10000 --batch_size 64 --lr 2e-4 \
  --round_budget_sec 300 --round_max_steps 4000 \
  --total_budget_sec 3300 --n_eval 64 --eval_seed 12345
python3 plots.py
```

Outputs land in `../../logs/` (`results.json`, `curves/distill_train.jsonl`,
`plots/*.png`). This is a CPU-only run (~25 min wall-clock for the full
32→1 cascade in this sandbox); no GPU required.

## RunPod full-scale rerun

`Dockerfile` / `requirements.txt` / `runpod_launch.sh` scaffold a full-scale
rerun (real UNet, CIFAR-10, much higher starting NFE) on an A100 pod — not
executed in this session. See `runpod_launch.sh` for the intended command
line and what needs to change to scale up.
