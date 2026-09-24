#!/usr/bin/env bash
# RunPod launch script: reruns the SAME MaskGIT mechanism (model.py's
# BidirectionalMaskedTransformer + cosine masking-ratio training objective +
# inference.py's confidence-ranked parallel decoding) at full paper scale on
# an A100-class GPU. This is scaffolding/config only -- not executed in the
# CPU sandbox that produced component-2-order-controller/logs/.
#
# What actually changes vs. the toy run (documented, not yet implemented here):
#   1. Tokenizer: 4-level pixel quantization (2 bits/pixel, no learned model)
#                 -> a real pretrained VQ-GAN (e.g. taming-transformers'
#                 ImageNet checkpoint), tokenizing 256x256 images into a
#                 16x16 grid of discrete codebook indices (vocab ~1024-8192).
#                 model.py's BidirectionalMaskedTransformer already only
#                 assumes a fixed sequence length + vocab size, so this is a
#                 drop-in swap of `data.py`'s tokenize/dequantize functions
#                 for the VQ-GAN's `encode`/`decode`.
#   2. Model:     ~325k params, 4 layers, d_model=96, seq_len=256
#                 -> MaskGIT's actual transformer (24 layers, d_model=768,
#                 ~24M+ params, seq_len=256 tokens over the VQ-GAN latent grid).
#   3. Data:      MNIST-16x16 (60k train images)
#                 -> ImageNet-1k (1.28M train images) via webdataset/streaming.
#   4. Steps:     8-step cosine unmasking schedule (unchanged -- this IS the
#                 mechanism under test; MaskGIT's paper also uses 8-12 steps).
#   5. Ablation:  add the Halton-scheduler order (arXiv 2503.17076) as a 4th
#                 arm alongside raster/random/confidence, matched step counts,
#                 for the follow-up comparison named in DECISION.md.
#   6. Compute:   4 CPU cores, ~30-45 min pretraining
#                 -> single A100, many hours, batch size scaled up (256-512).
#
# Usage (once the VQ-GAN/data/model-size swaps above are implemented):
#   docker build -t maskgit-fullscale .
#   docker run --gpus all -v $(pwd)/runpod_logs:/workspace/logs maskgit-fullscale
# or directly inside a RunPod pod with this repo checked out:
#   bash runpod_launch.sh

set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cuda}
BUDGET_SEC_TRAIN=${BUDGET_SEC_TRAIN:-28800}   # 8h
BATCH_SIZE=${BATCH_SIZE:-256}
STEPS=${STEPS:-8}

echo "== Full-scale MaskGIT launch (RunPod) =="
echo "This script currently reruns the toy-scale MNIST pipeline with scaled-up"
echo "hyperparameters as a smoke test of the launch path on a real GPU. Swap in"
echo "the VQ-GAN tokenizer / ImageNet data / full transformer size per the"
echo "TODOs above for an actual full-scale MaskGIT reproduction."
echo

python3 src/train_maskgit.py \
  --budget_sec "$BUDGET_SEC_TRAIN" --n_train 60000 --n_test 10000 --batch_size "$BATCH_SIZE"

python3 src/inference.py --n_eval 1000 --steps "$STEPS"

python3 src/plots.py

echo "Done. See logs/ for the training curve, arm-comparison results, and sample grids."
