# MaskGIT toy-scale reproduction — decoding-order controller

Faithful small-scale reproduction of MaskGIT's core mechanism (Chang et al.,
"MaskGIT: Masked Generative Image Transformer," CVPR 2022, arXiv 2202.04200):
a bidirectional transformer trained with a masked-token cross-entropy
objective under a cosine masking-ratio schedule, decoded at inference via
iterative parallel decoding — each step predicts all currently-masked
positions, ranks them by the model's own softmax confidence, and commits only
the top-confidence subset for that step's budget.

Design spec: `component-2-order-controller/DECISION.md`. Results and honest
write-up: `component-2-order-controller/logs/RUN_LOG.md`.

## Setup

```bash
cd component-2-order-controller/repro
pip install -r requirements.txt   # torch, torchvision, numpy, matplotlib (CPU is fine)
```

## Run (in order)

All scripts write logs to `../logs/` (curves as JSONL, results as JSON, plots
under `../logs/plots/`); the trained model checkpoint goes to `checkpoints/`.

```bash
cd src

# 1. Pretrain the bidirectional transformer on MNIST-16x16 (4-level-quantized
#    pixel tokens) with MaskGIT's cosine-masking-ratio masked-token objective.
#    Time-boxed; --budget_sec caps wall clock (30 min here, within the
#    30-45 min spec cap). Saves checkpoints/maskgit.pt.
python3 train_maskgit.py --budget_sec 1800 --n_train 20000 --n_test 2000 --batch_size 64

# 2. 3-arm decoding-order sweep: raster / random / confidence, all reusing the
#    SAME frozen trained model, same starting 75%-masked sequence, same
#    per-step reveal budget (MaskGIT's own cosine unmasking schedule) — only
#    reveal ORDER differs. Inference-only, fast (no training).
python3 inference.py --n_eval 1000 --steps 8

# 3. Plots: training curve, arms-comparison bar chart, sample reconstruction
#    grids (ground truth vs. masked input vs. each arm's reconstruction).
python3 plots.py
```

## Results at a glance

See `component-2-order-controller/logs/RUN_LOG.md` for the full narration,
exact hyperparameters, and wall-clock actually used, and
`component-2-order-controller/logs/results.json` for the headline per-arm
numbers. Plots (training curve, arms comparison, sample reconstruction grids)
are in `component-2-order-controller/logs/plots/`.

## What each file implements

- `src/data.py` — MNIST loading + downsampling to 16x16 + quantization to 4
  gray levels (2 bits/pixel); each of the 256 pixel positions becomes a token
  over vocab {0,1,2,3}, with a 5th MASK id for the training/inference objective.
  Self-contained (adapted from component-1's `repro/src/data.py`, same
  downsampling approach).
- `src/model.py` — `BidirectionalMaskedTransformer` (~325k params: 4 layers,
  d_model=96, 4 heads; token + positional embeddings over the 256-position
  sequence, no causal mask); `cosine_mask_ratio` and `sample_training_mask`
  implement MaskGIT's cosine random-masking-ratio training schedule.
- `src/train_maskgit.py` — masked-token cross-entropy training loop
  (loss computed only over masked positions), time-boxed by `--budget_sec`,
  logging loss/accuracy every N steps to `logs/curves/maskgit_train.jsonl`.
- `src/classifier.py` — the frozen MNIST-digit CNN architecture, copied from
  component-1 so this component is self-contained; weights are loaded
  read-only from component-1's `checkpoints/classifier.pt` (never retrained
  or modified here) as an independent secondary judge.
- `src/inference.py` — the 3-arm decoding-order sweep: `maskgit_reveal_schedule`
  (MaskGIT's cosine unmasking schedule, applied identically across arms) and
  `run_arm` (raster / random / confidence reveal order, all using the same
  model's own predictions to fill in committed tokens at each step). Computes
  final token-reconstruction accuracy and the classifier-judge score per arm
  on a shared fixed held-out eval set, writes `logs/results.json`.
- `src/plots.py` — training curve, arms-comparison bar chart, and sample
  reconstruction grids.

## Full-scale (RunPod) path

`Dockerfile`, `requirements-fullscale.txt`, and `runpod_launch.sh` scaffold
rerunning the same MaskGIT mechanism at full paper scale (a real VQ-GAN
tokenizer, ImageNet-scale MaskGIT transformer, and the Halton-scheduler
follow-up as a 4th ablation arm) on an A100 pod. See the comments at the top
of `runpod_launch.sh` for exactly what needs to be swapped in versus this toy
run. Not built or executed in this sandbox.
