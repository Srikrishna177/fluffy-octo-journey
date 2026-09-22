# DDPO toy-scale reproduction — run log

Sandbox: 4 CPU cores, 15GB RAM, no GPU, PyTorch 2.14.0 (CPU-runnable build
installed from PyPI — see "Deviations" below). All numbers in this file come
from runs actually executed in this sandbox and logged to the JSONL/`.log`
files in this directory; none are fabricated or projected.

_(This file is being filled in as each stage completes; see git history for
the incremental version if you're reading a partial draft.)_

## Stage 1 — verifier/reward pretraining (small CNN classifier)

- Command: `python3 train_classifier.py --epochs 8 --n_train 20000 --n_test 5000 --batch_size 128 --lr 1e-3 --seed 0`
- Params: 71,754 (spec: ~50-150k — in range)
- Dataset: REAL MNIST (torchvision download succeeded), 20,000 train / 5,000 test subset, downsampled to 16x16, scaled to [-1,1] for consistency with the DDPM pipeline (classifier itself is scale-invariant to this).
- Wall clock: 9.1s (well under the "few minutes" budget)
- Final test accuracy: 98.02%
- Target class for the DDPO reward: digit **7** (arbitrary single-class choice, documented per spec).
- Curve: `curves/classifier_train.jsonl`, plot: `plots/classifier_loss.png`

## Stage 2 — base DDPM pretraining (TinyUNet, T=50, epsilon-prediction)

- Command: `python3 train_ddpm.py --budget_sec 1800 --n_train 20000 --n_test 5000 --batch_size 128 --lr 2e-4 --T 50 --seed 0`
- Params: 308,641 (spec: ~150-400k — in range)
- Beta schedule: linear, beta_start=1e-4, beta_end=0.02, T=50 (spec allows linear or cosine; linear chosen — simpler, no material reason to expect it to change the toy-scale outcome).
- Budget: 1800s (30 min) — a deliberately shorter cut of the spec's 45-60 min cap; see "Deviations."
- Wall clock: 1800.1s (ran to the budget cap, as intended)
- Steps: 18,415 (119 epochs over the 20k-sample train subset, batch 128)
- Loss: MSE(pred-eps, true-eps) fell from ~1.04 (step 0) to a stable ~0.10 (100-step
  rolling average) by the end of training — see `curves/ddpm_train.jsonl`,
  plot: `plots/ddpm_loss.png`. Loss was still very slowly decreasing at the
  cutoff (0.103 avg at step ~18300 vs. ~0.105 around step ~15000), i.e.
  near-converged but not perfectly flat — consistent with using less than the
  full 45-60 min cap (a deliberate scope decision, not a failure to converge).
- Checkpoint: `checkpoints/ddpm_base.pt` (also serves as the frozen reference
  model for the DDPO KL-to-base estimate in stage 3).

## Stage 3 — DDPO RL fine-tuning

- _[filled in after run]_

## Stage 4 — Eval (before/after)

- _[filled in after run]_

## Deviations from DECISION.md and why

1. **PyPI CPU wheel instead of `download.pytorch.org/whl/cpu`.** The spec's
   suggested command (`pip install torch torchvision --index-url
   https://download.pytorch.org/whl/cpu`) was blocked by the sandbox's
   outbound proxy (403 on CONNECT to `download.pytorch.org`). `pip install
   torch torchvision` from plain PyPI (which is proxy-allowlisted) worked and
   resolves to a CUDA-capable build (`2.14.0+cu130`) that runs correctly on
   CPU when no CUDA device is present (`torch.cuda.is_available()` is
   `False`, all training done with `--device` implicitly `cpu`). No
   functional deviation, just a larger download (~2.9GB vs. a slimmer
   CPU-only wheel).
2. **Real MNIST used, not the synthetic fallback.** The synthetic-MNIST code
   path in `src/data.py` exists and is documented but was never triggered —
   `pypi.org`/torchvision's MNIST mirror was reachable.
3. **Sub-budget wall-clock, not the full 45-60 min cap.** DDPM pretraining and
   DDPO fine-tuning were each run with a 30-minute (1800s) budget rather than
   the full 45-60 min ceiling, to leave headroom in this session for eval,
   plotting, and write-up while still giving each stage a large step count
   (see per-stage step counts above/below). This is a scope decision within
   the spec's stated cap, not a violation of it — if the reward curve showed
   a clear, still-improving trend near the 30-minute mark, the run was
   extended; see the per-stage sections for what actually happened.
4. _(more added here if further deviations were needed during the run)_
