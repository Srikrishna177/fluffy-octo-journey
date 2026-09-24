# Component 2 — Difficulty-Conditioned Order Controller

*Reproduction + extension report. ~10-minute read. All numbers below are
copied or directly computed from `component-2-order-controller/logs/
results.json` and `component-2-order-controller/extension/logs/
results_4arm.json` — cross-checked against those files before writing this
report, not from memory or projection.*

## 1. The paper reproduced

**MaskGIT: Masked Generative Image Transformer** (Chang, Zhang, Jiang, Liu,
Freeman, Krishnan; CVPR 2022). MaskGIT decodes an image's discrete tokens over
~8-12 steps: at each step, a bidirectional transformer predicts *all*
currently-masked tokens in parallel, ranks them by the model's own softmax
confidence, and commits only the most-confident subset for that step. This
confidence-ranked reveal order is, by construction, a *difficulty-conditioned
order controller* — exactly the mechanism this program's proposal names.

**Toy-scale reproduction design** (`component-2-order-controller/DECISION.md`):
to isolate the effect of reveal *order* cleanly, we reframed the task as
masked-token reconstruction on MNIST rather than unconditional image
generation (which needs FID or a proxy quality judge). MNIST images are
downsampled to 16x16 and quantized to 4 gray levels (2 bits/pixel); each of
the 256 pixel positions is a "token" over a 4-symbol vocabulary plus a MASK
id. A small bidirectional transformer (4 layers, d_model=96, **324,772
params**) is trained once from scratch with MaskGIT's cosine masking-ratio
objective, then reused frozen by every decoding-order arm at inference — no
per-arm retraining, matching the paper's actual mechanism (one model, order
changes only at decode time).

## 2. Reproduction's real finding: confidence order did *not* win

Evaluated on **1,000 held-out MNIST test images**, each 75% masked (192/256
tokens), decoded over MaskGIT's own 8-step cosine reveal schedule, comparing
three arms that all use the *same* frozen model and only differ in which
positions get revealed at each step:

| arm | token-reconstruction accuracy | classifier-judge: top-1 = true digit |
|---|---|---|
| raster (fixed L-to-R) | **0.8499** | **50.2%** |
| random (fixed shuffle) | 0.8493 | 46.5% |
| **confidence (MaskGIT's mechanism)** | **0.8447** | **41.2%** |

This is reported plainly as a **negative result, not a bug**: MaskGIT's own
confidence-ranked decoding *underperformed* both fixed-order baselines at
this toy scale, on both the primary metric (token-reconstruction accuracy
against ground truth) and an independent secondary judge (component 1's
frozen MNIST classifier's confidence in / agreement with the true digit
label on the dequantized reconstruction). The gap is small in absolute terms
(~0.5 points on token accuracy) but consistent in direction across both
metrics, and larger and more decisive on the classifier-judge score (a
7-9 point gap). Sample reconstruction grids confirm this qualitatively:
confidence-order outputs are visibly more fragmented/incoherent than
raster's or random's.

The most likely explanation, flagged as a risk in `DECISION.md` *before* the
run (not a post-hoc excuse): the model's confidence is trained across the
*full* cosine masking-ratio schedule but evaluated at a fixed, unusually
high 75%-mask fraction with an unstructured mask shape — outside the bulk of
its training distribution, plausibly making its confidence miscalibrated
exactly where this experiment probes it. Full detail:
`component-2-order-controller/logs/RUN_LOG.md`.

## 3. Extension's question and design

If raw softmax confidence is a miscalibrated difficulty signal at this
regime, **can a separately-trained, ground-truth-supervised "learned" order
policy do better** — not just better than confidence, but well enough to
beat the fixed baselines too?

Design (`component-2-order-controller/extension/DECISION.md`):
1. Freeze the reproduction's transformer entirely (no retraining).
2. Generate supervision by running the frozen model on random 75%-masks of
   **20,000 MNIST train-split images** (the same pool the transformer itself
   trained on — fine to reuse, since only the frozen transformer's
   *training* needs a strict split, not this downstream policy). For every
   masked position, record a 99-dim feature vector (the model's own 96-dim
   hidden state at that position, plus normalized position index, max
   softmax confidence, and predictive entropy) and a binary label (did the
   model's argmax prediction match the true token).
3. Train a small MLP (99 -> 64 -> 32 -> 1) on this ~3.84M-row dataset to
   predict P(correct).
4. Add "learned" as a 4th arm to the **identical** eval protocol (same 1,000
   held-out test images, same seed 12345, same 75% starting mask, same
   8-step schedule) — only the within-step ranking changes, to the learned
   policy's predicted P(correct) instead of raw confidence.

**No train/eval leakage**: the policy's training data comes exclusively from
`torchvision`'s MNIST **train** split; the 1,000-image eval set comes
exclusively from the MNIST **test** split — disjoint datasets, not merely
disjoint indices. Verified explicitly in `extension/RUN_LOG.md`.

## 4. Extension's real result

| arm | token-recon acc | classifier: P(true digit) | classifier: top-1 = true digit |
|---|---|---|---|
| raster | **0.8499** | **0.4481** | **50.2%** |
| random | 0.8493 | 0.4224 | 46.5% |
| confidence | 0.8447 | 0.3710 | 41.2% |
| **learned** | 0.8452 | 0.3735 | 40.5% |

(raster/random/confidence were *recomputed*, not just copied, as a
determinism check — they matched the reproduction's original numbers
exactly, confirming the eval protocol was replicated bit-for-bit.)

The learned policy's own held-out validation accuracy at predicting
"is-correct" was **87.4%**, only ~2 points above a majority-class baseline
of **85.4%** (most positions are already predicted correctly by the base
model, so this task is naturally imbalanced) — i.e. even with real
ground-truth supervision and privileged access to the model's hidden state,
the "will this be correct" signal was only weakly separable at this scale.

**Which hypothesis this supports**: of the three outcomes
`extension/DECISION.md` pre-registered, the data supports **"the learned
policy doesn't clearly beat confidence either."** The token-recon accuracy
gap between learned and confidence (+0.0005) is roughly 0.3 standard errors
(SE ~ 0.0015) — indistinguishable from noise — and the secondary
classifier-judge metric actually *reverses* the sign (learned's top-1-match,
40.5%, is slightly *below* confidence's, 41.2%). Two metrics disagreeing on
direction is itself the signature of a near-zero true effect, not a real
edge either way. Both confidence-based arms remain clearly worse than the
fixed-order baselines (~2.7-3.1 SE below random and raster respectively).
Qualitatively, the "learned"-order sample reconstructions look similar to
"confidence"-order ones (both more fragmented than raster/random), not
closer to the fixed baselines. This is reported as-is, with no attempt to
force a cleaner story: **learned order-policy learning did not rescue
MaskGIT's confidence-ranking disadvantage at this toy scale.**

## 5. What this does and doesn't show at this scale

**What it shows**: at this toy scale (2-bit-quantized 16x16 MNIST, a
325k-parameter transformer, 20k training images), (a) confidence-ranked
decoding order can *underperform* trivial fixed orders, reproducing a real,
if narrow, negative result relative to MaskGIT's claimed advantage; and (b)
a small MLP given ground-truth supervision and richer input features than
raw confidence still could not build a meaningfully better difficulty
predictor, suggesting the bottleneck here is the *predictability* of
per-position correctness from pre-reveal information, not merely which
heuristic reads that information.

**What it doesn't show**: this is a 2-bit-per-pixel, no-VQ-GAN, 325k-param,
20k-image toy setup, nowhere near MaskGIT's original ImageNet/VQ-GAN-token
scale. It is entirely plausible that at full scale — a real learned
tokenizer with a much richer and more calibrated token vocabulary, a larger
transformer trained to convergence rather than a 30-minute-budget plateau,
and more masking-ratio diversity seen in training — both confidence-ranking
*and* a learned order policy behave differently (the original MaskGIT paper
does report a real ImageNet FID/IS advantage for confidence order). This
report makes no claim about that regime; it reports what was actually run
here.

**Path to full-scale validation**: `component-2-order-controller/repro/`
ships a RunPod-ready scaffold (`Dockerfile`, `requirements-fullscale.txt`,
`runpod_launch.sh`) for rerunning the same mechanism with a real VQ-GAN
tokenizer and an ImageNet-scale MaskGIT transformer on an A100 pod,
including the Halton-scheduler follow-up (arXiv 2503.17076) as a further
ablation arm. The extension's learned-policy code
(`component-2-order-controller/extension/src/`) is architecture-agnostic —
the same feature-extraction-plus-MLP recipe applies unchanged once a
full-scale frozen checkpoint exists; scaling it up is future work outside
this CPU-only sandbox, not executed here.

## References

- Full reproduction log: `component-2-order-controller/logs/RUN_LOG.md`
- Full extension log: `component-2-order-controller/extension/RUN_LOG.md`
- Reproduction numbers: `component-2-order-controller/logs/results.json`
- Extension numbers: `component-2-order-controller/extension/logs/results_4arm.json`
- Reproduction design: `component-2-order-controller/DECISION.md`
- Extension design: `component-2-order-controller/extension/DECISION.md`
