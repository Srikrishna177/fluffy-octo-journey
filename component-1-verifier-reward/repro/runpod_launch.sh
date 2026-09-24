#!/usr/bin/env bash
# RunPod launch script: reruns the SAME DDPO mechanism (src/diffusion.py's
# GaussianDiffusion.sample_trajectory + train_ddpo.py's Algorithm-1 update) at
# full paper scale on an A100-class GPU. This is scaffolding/config only — not
# executed in the CPU sandbox that produced component-1-verifier-reward/logs/.
#
# What actually changes vs. the toy run (documented, not yet implemented here):
#   1. T:        50  -> 1000                      (--T 1000)
#   2. UNet:     TinyUNet (~300k params, 16x16)    -> a real diffusers UNet2DModel
#                (or Stable-Diffusion's UNet if doing real text-to-image DDPO),
#                loaded via a new --arch diffusers flag in unet.py (TODO at
#                full-scale time: add a thin wrapper so train_ddpm.py /
#                train_ddpo.py can swap TinyUNet for UNet2DModel without other
#                code changes — p_mean_variance/sample_trajectory/log-prob math
#                in diffusion.py is architecture-agnostic already).
#   3. Data:     MNIST-16x16                       -> CIFAR-10 (32x32) for a
#                from-scratch unconditional run, or a pretrained SD checkpoint
#                + LAION-derived prompts for a real text-to-image DDPO run
#                matching the paper's actual setup.
#   4. Reward:   frozen 72k-param MNIST-digit CNN   -> ImageReward
#                (github.com/THUDM/ImageReward, installed via
#                requirements-fullscale.txt) or a CLIP-similarity reward, both
#                black-box/non-differentiable exactly as DDPO's method requires.
#   5. Compute:  4 CPU cores, ~30-45 min/stage      -> single A100, hours/stage,
#                batch size scaled up (256-512 trajectories/iteration is typical
#                in the reference kvablack/ddpo-pytorch implementation).
#
# Usage (once the arch/data/reward swaps above are implemented):
#   docker build -t ddpo-fullscale .
#   docker run --gpus all -v $(pwd)/runpod_logs:/workspace/logs ddpo-fullscale
# or directly inside a RunPod pod with this repo checked out:
#   bash runpod_launch.sh

set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cuda}
T=${T:-1000}
BUDGET_SEC_DDPM=${BUDGET_SEC_DDPM:-10800}   # 3h
BUDGET_SEC_DDPO=${BUDGET_SEC_DDPO:-10800}   # 3h
BATCH_SIZE=${BATCH_SIZE:-256}

echo "== Full-scale DDPO launch (RunPod) =="
echo "This script currently reruns the toy-scale MNIST pipeline with scaled-up"
echo "hyperparameters as a smoke test of the launch path on a real GPU. Swap in"
echo "the diffusers UNet / CIFAR-or-SD data / ImageReward reward per the TODOs"
echo "above for an actual full-scale DDPO reproduction."
echo

python3 src/train_classifier.py \
  --epochs 10 --n_train 60000 --n_test 10000 --batch_size 256

python3 src/train_ddpm.py \
  --budget_sec "$BUDGET_SEC_DDPM" --n_train 60000 --n_test 10000 \
  --batch_size "$BATCH_SIZE" --T "$T"

python3 src/train_ddpo.py \
  --budget_sec "$BUDGET_SEC_DDPO" --batch_size 64 --T "$T"

python3 src/eval.py --n_eval 256 --T "$T"

echo "Done. See logs/ for curves, eval_before_after.json, and sample grids."
