# Component 2 extension — learned order policy

## What this tests

The reproduction (`../logs/RUN_LOG.md`) found that MaskGIT's confidence-ranked
reveal order **underperformed** fixed baselines (raster, random) on toy-scale
MNIST-16x16 masked-token reconstruction. The suspected cause: raw softmax
confidence is a miscalibrated difficulty signal at the 75%-mask regime, which
is outside the bulk of the model's cosine-schedule training distribution.

This extension asks: **can a separately-trained, ground-truth-supervised
"learned" difficulty predictor do better than raw confidence** — and does
ordering by it beat not just confidence, but the fixed baselines too?

See `DECISION.md` for the full design and pre-registered hypotheses.

## Result (one line)

**No.** The learned policy performs statistically indistinguishably from raw
confidence (and, on the secondary classifier-judge metric, slightly worse),
and both remain clearly below the raster/random fixed baselines. Full table
and analysis: `RUN_LOG.md`.

## How to rerun

All commands assume `cd extension/src`. Requires `../../repro/checkpoints/
maskgit.pt` (the frozen base transformer) and
`../../../component-1-verifier-reward/repro/checkpoints/classifier.pt` (the
frozen classifier-judge) to already exist.

```bash
# Stage 1: generate (feature, is_correct) supervision from the MNIST TRAIN split
python3 gen_policy_data.py
# -> ../data/policy_dataset.pt

# Stage 2: train the policy MLP
python3 train_policy.py
# -> ../checkpoints/policy.pt, ../logs/curves/policy_train.jsonl

# Stage 3: 4-arm eval sweep (raster/random/confidence recomputed + learned, on the MNIST TEST split)
python3 inference_ext.py
# -> ../logs/results_4arm.json, ../logs/sample_recon_data_4arm.pt

# Plots
python3 plots_ext.py
# -> ../logs/plots/{policy_training_curve,arms_comparison_4arm,sample_reconstructions_4arm}.png
```

Total wall clock in this sandbox (4 CPU cores, no GPU): ≈7.5 minutes.

## Files

- `src/policy_common.py` — shared feature extraction (`forward_with_hidden`,
  `extract_features`) and the `PolicyMLP` module, used identically at
  training-data-generation time and at 4th-arm inference time.
- `src/gen_policy_data.py` — Stage 1.
- `src/train_policy.py` — Stage 2.
- `src/inference_ext.py` — Stage 3 (imports `maskgit_reveal_schedule` and
  `run_arm` directly from `../../repro/src/inference.py` to guarantee an
  identical eval protocol to the reproduction).
- `src/plots_ext.py` — plots.
- `data/policy_dataset.pt` — Stage 1 output (not committed if large; regenerate via the command above).
- `checkpoints/policy.pt` — trained policy weights.
- `logs/` — curves, `results_4arm.json`, sample-reconstruction tensors, plots.
- `DECISION.md` — extension design (written before this run).
- `RUN_LOG.md` — exact commands, hyperparameters, full results table, honest
  hypothesis verdict.
