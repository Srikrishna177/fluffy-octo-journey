# Component 2 extension — learned order policy — run log

Sandbox: 4 CPU cores, 15GB RAM, no GPU, PyTorch 2.14.0+cu130 (already installed).
All numbers below come from runs actually executed in this sandbox and logged
to the JSON/JSONL files in `extension/logs/`; none are fabricated or projected.
Frozen base transformer: `../repro/checkpoints/maskgit.pt` (325k params, trained
step=1318) — its weights are never modified anywhere in this extension.

## Stage 1 — policy training-data generation

- Command: `python3 gen_policy_data.py` (defaults: `--n_train_images 20000
  --mask_frac 0.75 --batch_size 500 --seed 999`)
- Data: the identical 20,000-image MNIST **train**-split pool the base
  transformer was pretrained on, reproduced deterministically by calling
  `load_mnist16_tokens(n_train=20000, n_test=2, seed=0)` — the same
  `(n_train, seed)` as `train_maskgit.py`'s run. (`n_test` does not affect
  which images land in `train_tokens`: the train permutation is drawn from
  `torchvision`'s MNIST **train** split before `n_test` is ever consulted —
  verified by reading `data.py`'s `load_mnist16_tokens`.)
- For each of the 20,000 images, one random 75%-masked view is sampled
  (seed=999, independent of the base-training seed=0 and the eval seed=12345)
  and run once through the frozen model (`eval()`, `no_grad`, weights
  untouched). For every masked position, recorded:
  - **feature vector** (99-dim; `extension/src/policy_common.py`):
    the model's own 96-dim post-encoder/post-LayerNorm hidden state at that
    position (the exact vector its own output `head` sees) + normalized
    position index (1) + max softmax confidence (1, logit-derived) +
    predictive entropy (1, logit-derived).
  - **binary label**: 1.0 if the frozen model's argmax prediction at that
    position equals the true quantized pixel token, else 0.0.
- Output: `extension/data/policy_dataset.pt` — **3,840,000 rows** (20,000
  images x 192 masked positions), positive rate (base model already correct)
  = **0.8550**.
- **Wall clock: 65.7s** (compute) / 74.0s total incl. imports/MNIST load.

## Stage 2 — policy MLP training

- Command: `python3 train_policy.py` (defaults: `--epochs 8 --batch_size 4096
  --lr 1e-3 --val_frac 0.1 --seed 1234`)
- Model: `PolicyMLP` — 99 -> 64 -> 32 -> 1 (ReLU, `BCEWithLogitsLoss` on the
  raw logit), Adam, no weight decay. 90/10 train/val split of the Stage-1
  dataset (3,456,000 train / 384,000 val rows) — this split is only to
  monitor the policy's own fit and is separate from the eval-pool leakage
  question addressed below.
- Curve: `extension/logs/curves/policy_train.jsonl`, plot:
  `extension/logs/plots/policy_training_curve.png`.
- Result after 8 epochs: **train_acc 0.8753, val_acc 0.8741** vs. a
  **val majority-class baseline of 0.8541** (always predicting "correct",
  since positives are 85.5% of the data). The policy improves on the
  majority baseline by only **~2 percentage points** — it has learned some
  real signal (it is not just parroting the base rate), but the "will this
  position be revealed correctly" target is only weakly predictable from the
  frozen model's own hidden state / logit features at this toy scale. This
  is logged honestly here because it directly foreshadows the 4-arm result
  below: a policy with only modest discriminative power is not expected to
  produce a much better *reveal ranking* than raw confidence.
- **Wall clock: 269.0s** (8 epochs over 3.84M rows on 4 CPU cores) / 274.8s
  total incl. data load.

## Stage 3 — 4-arm decoding-order sweep

- Command: `python3 inference_ext.py` (defaults: `--n_eval 1000
  --n_test_pool 2000 --mask_frac 0.75 --steps 8 --seed 12345`) — **identical**
  arguments/defaults to the reproduction's `repro/src/inference.py`.
- Eval set, initial mask, and reveal-budget schedule are **recomputed by the
  identical code path** (`load_mnist16_tokens(n_train=2, n_test=2000,
  seed=0)` for the 1000-image held-out MNIST **test**-split eval pool, same
  seed=12345 75%-mask construction, same `maskgit_reveal_schedule`) —
  imported directly from `repro/src/inference.py`, not re-derived, so there
  is no chance of protocol drift between reproduction and extension.
- **Determinism / leakage sanity check**: raster, random, and confidence were
  **recomputed** (not copied) in this same script and compared to the
  original `../logs/results.json`. All three matched **exactly**
  (`exact_match: true` in `extension/logs/results_4arm.json`, e.g. raster
  0.8499270833333333 both times) — confirming the eval protocol was
  reproduced bit-for-bit identically (deterministic: `eval()` mode, no
  dropout, fully seeded RNGs) before trusting the new 4th arm's number.
- 4th arm ("learned"): identical mechanics to the "confidence" arm in
  `run_arm` (same schedule, same commit-the-model's-own-argmax-prediction
  rule) except the per-step ranking score is `sigmoid(policy(features))`
  computed fresh at every step from the current partially-revealed sequence,
  instead of raw softmax confidence.
- **Wall clock: 94.1s** for the full 4-arm sweep (1000 images, 8 steps).

### Results (`extension/logs/results_4arm.json`, plots: `extension/logs/plots/arms_comparison_4arm.png`, `sample_reconstructions_4arm.png`)

| arm | token-recon acc | std across images | classifier: P(true digit) | classifier: top-1 = true digit |
|---|---|---|---|---|
| raster | **0.8499** | 0.0440 | **0.4481** | **50.2%** |
| random | 0.8493 | 0.0434 | 0.4224 | 46.5% |
| confidence | 0.8447 | 0.0463 | 0.3710 | 41.2% |
| **learned** | 0.8452 | 0.0459 | 0.3735 | 40.5% |

Standard error of the mean token-recon accuracy is ≈0.046/√1000 ≈ 0.0015.

- **learned vs. confidence**: +0.0005 token-recon acc (≈0.3 SE — not
  distinguishable from noise), but the secondary classifier-judge metric
  actually *flips*: learned's top-1-match (40.5%) is slightly *below*
  confidence's (41.2%). The two signs disagree across the two metrics, which
  is itself evidence the true gap is ≈0.
- **learned vs. random**: −0.0041 token-recon acc (≈2.7 SE) — learned is
  clearly still worse than the random-order fixed baseline.
- **learned vs. raster**: −0.0047 token-recon acc (≈3.1 SE) — learned is
  clearly still worse than the raster-order fixed baseline.

### Which hypothesis this supports

Per `extension/DECISION.md`'s three pre-registered outcomes, the data
supports **outcome 3: "the learned policy doesn't clearly beat confidence
either."** The ground-truth-supervised policy performs statistically
indistinguishably from raw softmax confidence on the primary metric (a
0.0005 gap, far under 1 SE) and, if anything, slightly worse on the
secondary classifier-judge metric — the two metrics disagree on which of
{confidence, learned} is better, which is itself the signature of a
near-zero true difference, not a real advantage in either direction. Both
confidence-based arms remain clearly worse than the two fixed-order
baselines (raster, random) on both metrics. This is **not** a forced or
ambiguous read: the numbers point in one consistent direction (learned ≈
confidence < random ≈ raster), it is simply not the direction the extension
hoped to find.

The Stage 2 result (policy val accuracy only ~2 points above the majority
baseline) is the most plausible mechanism: even with real ground-truth
supervision and access to the model's own hidden state (strictly more
information than raw softmax confidence has), predicting "will this
argmax prediction be correct" from features available *before* revealing
is close to the ceiling of what this toy-scale model's representations
support — there just isn't much more calibrated signal to extract than raw
confidence already captures. This also matches the qualitative check: the
"learned"-order sample reconstructions (`sample_reconstructions_4arm.png`)
look visually similar to "confidence"-order ones (both more fragmented than
raster/random), not closer to the fixed-baseline reconstructions.

## Deviations from `extension/DECISION.md` and why

1. **Feature choice made concrete** (DECISION.md left this open: "hidden
   state / logit-derived features plus a positional embedding/index, your
   choice"). Used: 96-dim post-encoder/post-LayerNorm hidden state (the
   vector the model's own output head consumes) + normalized position index
   + max softmax confidence + predictive entropy = 99 dims. Confidence and
   entropy were included alongside the hidden state (not just the hidden
   state alone) specifically to give the policy *at least* as much
   information as the "confidence" arm has, so a null result cannot be
   blamed on withholding that signal from the learned policy.
2. **Training data uses one random 75%-mask per image (single-step
   supervision), not a simulated multi-step decode trajectory.** This
   follows the literal task instructions ("run it on random 75%-masks...for
   each masked position in each sampled masking, record...") and keeps the
   supervision distribution matched to the eval's *starting* condition. A
   caveat, stated plainly: at later decode steps the sequence is only
   partially masked (fewer than 192 positions), so the policy is evaluated
   in step 2..8 on a somewhat different input distribution (mixed
   revealed/masked context) than it was trained on (always the initial 75%
   mask). This mismatch is a plausible additional contributor to the learned
   policy's flat performance and is disclosed here rather than glossed over.
3. **raster/random/confidence were recomputed** in `inference_ext.py` rather
   than only copying `../logs/results.json`'s numbers, specifically to get a
   bit-for-bit determinism check for free (see Stage 3) — a stronger
   guarantee than trusting that two separately-written scripts implement
   "the exact same protocol" by inspection alone. Both are reported in
   `results_4arm.json` (the copied numbers agree with the recomputed ones
   exactly, so there is only one set of numbers in the final table).
4. **No mid-run hyperparameter changes.** The policy MLP architecture (2
   hidden layers, 64/32 units), learning rate, and epoch count were fixed
   before Stage 3 was run and were not adjusted after seeing the 4-arm
   result.

## Leakage sanity check (explicit, as required)

The 1000 eval-pool images used for the 4-arm comparison come from
`torchvision.datasets.MNIST(..., train=False)` (the MNIST **test** split).
The policy's training data (Stage 1) comes exclusively from
`torchvision.datasets.MNIST(..., train=True)` (the MNIST **train** split) —
a disjoint underlying dataset object, not merely a disjoint index range
within the same pool. The policy checkpoint (`extension/checkpoints/
policy.pt`) is trained once in Stage 2 and never updated or selected using
any information from the Stage-3 eval run. **No train/eval leakage.**

## Wall-clock summary

| stage | command | wall clock |
|---|---|---|
| policy data generation | `python3 gen_policy_data.py` | 74.0s |
| policy MLP training | `python3 train_policy.py` | 274.8s |
| 4-arm eval sweep | `python3 inference_ext.py` | 96.0s |
| plots | `python3 plots_ext.py` | ~5s |
| **total compute** | | **≈7.5 min**, well inside the ~30-45 min task time-box (most of the session's time went to reading the existing code, implementation, and this write-up, as intended). |
