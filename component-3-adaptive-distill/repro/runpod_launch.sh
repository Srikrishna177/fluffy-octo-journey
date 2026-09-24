#!/usr/bin/env bash
# Full-scale rerun launcher for RunPod (A100-class GPU).
#
# This toy repro (component-3-adaptive-distill/repro/src/) used a ~309k-param
# TinyUNet on MNIST-16x16 with a T=50 base schedule subsampled to 32 DDIM
# steps. The intended full-scale rerun, NOT executed in this CPU sandbox:
#
#   - Real UNet (e.g. a small Stable-Diffusion-style UNet, tens of millions
#     of params) instead of TinyUNet.
#   - CIFAR-10 (paper's own setting) or a downsampled ImageNet, instead of
#     MNIST-16x16.
#   - Base teacher trained (or a public pretrained DDPM checkpoint reused)
#     with a much higher-resolution schedule (paper uses up to T=1024/4000);
#     start the halving cascade from e.g. 256 or 512 steps instead of 32, so
#     the cascade has room to show the paper's actual claimed result (1-4
#     step samples close to the many-step teacher) before running into a
#     toy-capacity floor the way this CPU repro did.
#   - Same progressive-halving mechanism and metrics (trajectory pixel/LPIPS
#     distance to teacher, FID/IS as the "classifier-judge" analogue, NFE and
#     wall-clock) -- src/distill.py's core loop generalizes directly; only
#     the model, dataset and schedule length need to change.
#
# Usage (after `docker build` from this directory, or directly on a RunPod
# pod with this repo checked out and requirements.txt installed):
#
#   ./runpod_launch.sh
#
# Edit the flags below once the full-scale model/data code lands; kept as a
# thin wrapper over the same distill.py CLI shape as the toy run for parity.

set -euo pipefail
cd "$(dirname "$0")"

python3 src/distill.py \
  --T 1000 \
  --base_nfe 256 \
  --n_train 50000 \
  --batch_size 128 \
  --lr 2e-4 \
  --round_budget_sec 1800 \
  --round_max_steps 20000 \
  --total_budget_sec 43200 \
  --n_eval 256 \
  --eval_seed 12345
