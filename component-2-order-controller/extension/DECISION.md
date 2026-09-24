# Component 2 — extension design

## Recap of reproduction result

MaskGIT's confidence-ranked reveal order did **not** beat fixed baselines at toy scale: on
1000 held-out MNIST-16x16 images (75% masked, 8-step decode, one shared frozen transformer),
raster order (0.8499 token-recon acc) and random order (0.8493) both beat confidence order
(0.8447), and the gap is larger and same-direction on the independent classifier-judge metric
(50.2%/46.5% vs. 41.2% top-1-match). Sample grids confirm this qualitatively: confidence-order
reconstructions are visibly more fragmented. Likely cause (flagged in the reproduction's
`DECISION.md` in advance, not post-hoc): the model's raw softmax confidence is a miscalibrated
difficulty signal at the 75%-mask extreme, which sits outside the bulk of its cosine-schedule
training distribution. Full detail: `../logs/RUN_LOG.md`.

## The question this extension tests

**If raw softmax confidence is a miscalibrated difficulty signal, can a separately-trained,
*learned* order policy produce a better-calibrated signal — and does ordering by it beat not
just the confidence heuristic, but also the fixed (random/raster) baselines?**

This is sharper than "does a learned order help" in the abstract: it directly targets the
specific failure diagnosed in the reproduction, rather than a generic add-on.

## Design

1. **Freeze everything from the reproduction.** Reuse `../repro/checkpoints/maskgit.pt`
   as-is — no retraining of the base transformer. Only the reveal-order mechanism changes,
   same principle Component 1's extension used (isolate the one variable under test).
2. **Learned policy = a small, separately-supervised correctness predictor.** For each
   masked position, the policy predicts P(the frozen transformer's argmax prediction at this
   position will be correct), using the transformer's own per-position hidden state /
   logit-derived features plus a positional embedding as input to a small MLP (1-2 hidden
   layers). This is "difficulty-conditioned" (predicts easy-vs-hard) and "learned" (trained
   with real supervision on whether predictions are actually correct, not just reading off
   the model's own possibly-miscalibrated softmax) — directly addressing the diagnosed
   failure mode rather than re-deriving another heuristic.
3. **Training data, leak-free.** Generate (feature, is_correct) pairs by running the frozen
   transformer on random 75%-masks of images from the **MNIST train split** (the 20,000
   images the base transformer itself trained on — reusing them for policy supervision is
   fine since the base transformer is now frozen and this isn't the model being evaluated).
   The reproduction's comparison arms were evaluated exclusively on a **test-pool** eval set;
   the policy never sees or is tuned against those images, so the eventual 4-arm comparison
   stays uncontaminated.
4. **Inference**: same protocol as the reproduction (same 1000-image eval pool, same seed,
   same 75% starting mask, same 8-step reveal-budget schedule) — only the ranking-within-step
   changes to the learned policy's predicted P(correct), highest first.
5. **Comparison**: add this 4th arm ("learned") to the existing raster/random/confidence
   table from the reproduction, same two metrics (token-recon accuracy, classifier-judge
   score), same eval images — a direct, apples-to-apples extension of the existing table.

## Hypothesis and honest expectation

No forced narrative. Plausible outcomes:
- Learned policy beats confidence AND matches/beats the fixed baselines → supports "a
  properly-calibrated difficulty signal is genuinely useful; the paper's heuristic just
  wasn't well-calibrated for this regime."
- Learned policy beats confidence but still underperforms fixed baselines → supports "order
  itself isn't the lever at this scale/task; something else (e.g. spatial locality that
  raster/random incidentally preserve) matters more than difficulty-based prioritization."
- Learned policy doesn't clearly beat confidence either → supports "the correctness signal
  is inherently hard to predict ahead of revealing, regardless of who computes it" — also a
  real, useful finding about the limits of any order controller on this toy task.
Whichever happens gets reported plainly, per the program's honesty rule.

## Time-box

Policy training data generation + MLP training: cheap (no need to touch the frozen
transformer's weights), budget ~15-20 min wall-clock total. Inference sweep for the 4th arm:
same cost as the reproduction's sweep (~1 minute per 1000 images), negligible. Total task
budget: aim to wrap up within ~30-45 minutes.
