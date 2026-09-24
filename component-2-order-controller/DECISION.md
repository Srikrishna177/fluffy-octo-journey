# Component 2 — paper selection decision

## Candidates considered (scout pass, 2026-09-24)

1. **MaskGIT** — "Masked Generative Image Transformer" (Chang, Zhang, Jiang, Liu, Freeman,
   Krishnan; CVPR 2022; arXiv 2202.04200). Bidirectional transformer over image tokens,
   decoded over ~8-12 steps: at each step, predicts all masked tokens in parallel, ranks
   them by the model's own confidence, and commits only the most-confident subset —
   confidence-ranked reveal order is a difficulty-conditioned order controller by
   construction. Multiple public PyTorch reimplementations, trainable from scratch at toy
   scale. The Halton-scheduler follow-up (arXiv 2503.17076, ICLR 2025) supplies a ready-made
   ablation template (confidence vs. random vs. structured order, matched step counts).
2. **Adaptive Order Policies for Masked Diffusion** (Mohamud, Hasan, Ravanelli, Bengio; 2026;
   arXiv 2606.00295). Trains a small auxiliary policy network to choose unmasking order via a
   reweighted diffusion loss — an explicitly *learned* order controller, benchmarked on
   Sudoku. No public code found; would be a from-scratch reimplementation.
3. **Learning Unmasking Policies for Diffusion Language Models** (Apple; 2025; arXiv
   2512.09106, code public). RL-trained policy over a frozen pretrained diffusion LM's
   per-token confidences. Public code, but needs a LLaDA/Dream-scale pretrained base model as
   the RL environment — not CPU-feasible at toy scale without a two-stage from-scratch
   pipeline (pretrain a tiny masked-diffusion LM, then RL the policy on top).

Full scout report preserved in the session log; not duplicated here.

## Decision: anchor reproduction on MaskGIT, extension on the "learned order policy" idea

**Reproduction anchor: MaskGIT** (candidate 1). Most de-risked: mature code, single-stage
training from scratch at toy scale on CPU, and the confidence-ranked reveal mechanism is a
textbook difficulty-conditioned order controller — reproducing "does confidence order beat
fixed order" is close to a direct fork-and-run once toy-scaled.

**Extension: a small *learned* order-policy** (inspired by candidate 2's framing — order as a
learnable component rather than a fixed heuristic — reimplemented at toy scale on top of the
same trained MaskGIT-style model rather than a separate from-scratch Sudoku pipeline, to reuse
infrastructure and reduce risk the way Component 1's extension reused its reproduction's
checkpoints). This directly tests the proposal's literal question: does a *learned* order
controller beat both a *hand-designed* difficulty heuristic (confidence-ranking) and the fixed
baselines (random, raster/left-to-right) — the full spectrum the proposal names.

## Toy-scale design (CPU-only sandbox, no GPU)

- **Task reframing for a clean, direct metric**: rather than unconditional generation (which
  needs FID or a proxy quality judge), use **masked-token reconstruction/inpainting**: start
  from an image with a large fraction of tokens masked (e.g. 75%), progressively unmask over a
  fixed step budget, and measure final **token-reconstruction accuracy against the known
  ground truth**. This isolates the effect of *order* cleanly (same trained model, same masking
  ratio, same step budget — only the order in which tokens are revealed differs across arms),
  avoiding the confound of also needing a separate quality metric.
- **Tokenization**: to avoid needing a separately-trained VQ-GAN tokenizer (not toy-scale
  feasible to train well from scratch), quantize MNIST-16x16 pixels directly to a small number
  of gray levels (4 levels = 2 bits/pixel) and treat each of the 256 pixel positions as a
  "token" over a vocabulary of 4 symbols + 1 mask token. This is a documented, defensible
  simplification: the scientific question under test is about *order*, not tokenizer quality.
- **Base model**: one small bidirectional transformer (~4 layers, small width, a few hundred
  K params), trained once from scratch with MaskGIT's cosine random-masking-ratio objective
  (standard masked-token cross-entropy). Trained once; all order-controller arms below reuse
  this single frozen model at inference/decoding time — no separate training run needed per
  arm, keeping this component's compute budget much lighter than Component 1's RL runs.
- **Arms compared** (same model, same step budget, same starting mask):
  1. **Raster order** (fixed left-to-right, top-to-bottom) — the strict-order baseline.
  2. **Random order** (uniformly shuffled reveal order) — the arbitrary-order baseline.
  3. **Confidence order** (MaskGIT's own mechanism: reveal highest-confidence masked tokens
     first each step) — the hand-designed difficulty-conditioned heuristic.
  4. **Learned order policy** (extension, built after 1-3 are verified) — a small auxiliary
     network trained to predict a reveal-priority score per masked position, optimized (via a
     reweighted reconstruction loss, in the spirit of arXiv 2606.00295) to beat arm 3, not
     just arms 1-2.
- **Metric**: final token-reconstruction accuracy (and, as a secondary/qualitative check,
  reuse Component 1's frozen MNIST classifier checkpoint as an independent "does this still
  look like the right digit" judge on the dequantized reconstructions).
- **Time-box**: base transformer pretraining capped at ~30-45 min wall-clock. Arms 1-3 are
  inference-only sweeps over the same trained model (should be fast, minutes, not
  training-bounded). The learned-policy extension gets its own time-box decided after the
  reproduction (arms 1-3) lands, mirroring how Component 1 phased reproduction before
  extension.

## What ships to RunPod later

`Dockerfile` / `requirements.txt` / `runpod_launch.sh` under `repro/` running the identical
mechanism at full scale (a real VQ-GAN tokenizer, ImageNet-scale MaskGIT, the Halton-scheduler
comparison) on an A100 pod. Not executed in this session.
