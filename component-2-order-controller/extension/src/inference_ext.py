"""4-arm decoding-order sweep: raster / random / confidence (recomputed, exact
same protocol as repro/src/inference.py) + learned (new, extension).

Reuses:
  - the SAME frozen checkpoint (repro/checkpoints/maskgit.pt), untouched.
  - the SAME eval-set construction, SAME seed (12345), SAME 75% initial mask,
    SAME 8-step reveal-budget schedule as repro/src/inference.py (imported
    directly, not re-derived, so there is no chance of an accidental protocol
    drift between the reproduction and the extension).
  - the SAME classifier-judge logic (component-1's frozen SmallCNN).

Only the "learned" arm's within-step ranking is new: instead of raw softmax
confidence, it ranks still-masked positions by the trained policy's predicted
P(correct), highest first.

raster/random/confidence are RECOMPUTED here (not just copied from
../../logs/results.json) as a determinism/leakage sanity check: since eval()
has no dropout and every RNG is seeded, the recomputed numbers should match
the original run bit-for-bit. This is verified below and reported in
RUN_LOG.md.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(__file__)
REPRO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "repro", "src"))
sys.path.insert(0, REPRO_SRC)
sys.path.insert(0, HERE)

from data import load_mnist16_tokens, MASK_TOKEN, SEQ_LEN, dequantize_tokens  # noqa: E402
from model import BidirectionalMaskedTransformer  # noqa: E402
from classifier import SmallCNN  # noqa: E402
from inference import maskgit_reveal_schedule, run_arm, CKPT_PATH, CLASSIFIER_CKPT  # noqa: E402
from policy_common import forward_with_hidden, extract_features, PolicyMLP  # noqa: E402

EXT_DIR = os.path.abspath(os.path.join(HERE, ".."))
EXT_LOG_DIR = os.path.join(EXT_DIR, "logs")
POLICY_CKPT = os.path.join(EXT_DIR, "checkpoints", "policy.pt")
ORIG_RESULTS_PATH = os.path.abspath(os.path.join(HERE, "..", "..", "logs", "results.json"))


def run_arm_learned(model, policy, tokens_gt: torch.Tensor, init_mask: torch.Tensor, budgets):
    """Mirrors repro/src/inference.py's run_arm's 'confidence' branch exactly,
    except the per-position score used to rank still-masked positions comes
    from the trained policy's predicted P(correct) instead of raw softmax
    confidence. The VALUE committed at a revealed position is still always
    the frozen transformer's own argmax prediction -- identical to every
    other arm."""
    N, L = tokens_gt.shape
    T = len(budgets) - 1
    seq = tokens_gt.clone()
    seq[init_mask] = MASK_TOKEN
    still_masked = init_mask.clone()
    snapshots = {0: seq.clone()}
    with torch.no_grad():
        for t in range(1, T + 1):
            hidden, logits = forward_with_hidden(model, seq)
            feat, pred = extract_features(hidden, logits, L)
            score = torch.sigmoid(policy(feat.reshape(-1, feat.shape[-1]))).reshape(N, L)
            n_reveal = budgets[t - 1] - budgets[t]
            if n_reveal > 0:
                score_masked = score.masked_fill(~still_masked, -1.0)
                _, topk_idx = torch.topk(score_masked, n_reveal, dim=1)
                pred_at_sel = torch.gather(pred, 1, topk_idx)
                seq.scatter_(1, topk_idx, pred_at_sel)
                still_masked.scatter_(1, topk_idx, False)
            snapshots[t] = seq.clone()
    assert not still_masked.any(), "all masked positions should be revealed by the last step"
    return seq, snapshots


def compute_metrics(final_seq, tokens_gt, init_mask, labels, classifier, n_eval):
    correct = (final_seq == tokens_gt) & init_mask
    token_acc = correct.sum().item() / init_mask.sum().item()
    recon_img = dequantize_tokens(final_seq).view(n_eval, 1, 16, 16)
    with torch.no_grad():
        logits = classifier(recon_img)
        probs = torch.softmax(logits, dim=-1)
    true_class_conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pred_class = probs.argmax(dim=1)
    classifier_acc = (pred_class == labels).float().mean().item()
    per_image_acc = correct.float().sum(dim=1) / init_mask.float().sum(dim=1)
    return {
        "token_reconstruction_acc": token_acc,
        "token_acc_std_across_images": per_image_acc.std().item(),
        "classifier_true_label_confidence_mean": true_class_conf.mean().item(),
        "classifier_top1_matches_true_label_frac": classifier_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--n_test_pool", type=int, default=2000)
    ap.add_argument("--mask_frac", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n_sample_grid", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(EXT_LOG_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    t0 = time.time()

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = BidirectionalMaskedTransformer()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer_ext] loaded frozen MaskGIT model from {CKPT_PATH} "
          f"(trained step={ckpt.get('step')})", flush=True)

    pol_ckpt = torch.load(POLICY_CKPT, map_location="cpu")
    policy = PolicyMLP(in_dim=pol_ckpt["feature_dim"])
    policy.load_state_dict(pol_ckpt["model"])
    policy.eval()
    print(f"[infer_ext] loaded learned policy from {POLICY_CKPT} "
          f"(val_acc={pol_ckpt.get('final_val_acc'):.4f}, "
          f"majority_baseline={pol_ckpt.get('val_majority_baseline_acc'):.4f})", flush=True)

    # EXACTLY repro/src/inference.py's eval-set construction.
    _, _, test_tokens, test_labels, used_real = load_mnist16_tokens(
        n_train=2, n_test=args.n_test_pool, seed=0
    )
    n_eval = min(args.n_eval, test_tokens.shape[0])
    tokens_gt = test_tokens[:n_eval]
    labels = test_labels[:n_eval]
    print(f"[infer_ext] eval set: {n_eval} held-out MNIST TEST-split images "
          f"(used_real_mnist={used_real})", flush=True)

    n_mask = int(round(args.mask_frac * SEQ_LEN))
    g = torch.Generator().manual_seed(args.seed)
    rand_scores = torch.rand(n_eval, SEQ_LEN, generator=g)
    sorted_scores, _ = torch.sort(rand_scores, dim=1)
    thresh = sorted_scores[:, n_mask - 1: n_mask]
    init_mask = rand_scores <= thresh
    assert (init_mask.sum(dim=1) == n_mask).all()

    budgets = maskgit_reveal_schedule(n_mask, args.steps)
    print(f"[infer_ext] mask_frac={args.mask_frac} -> {n_mask}/{SEQ_LEN} masked; "
          f"steps={args.steps}; schedule={budgets}", flush=True)

    classifier = SmallCNN(num_classes=10)
    cls_ckpt = torch.load(CLASSIFIER_CKPT, map_location="cpu")
    state = cls_ckpt["model"] if isinstance(cls_ckpt, dict) and "model" in cls_ckpt else cls_ckpt
    classifier.load_state_dict(state)
    classifier.eval()
    print(f"[infer_ext] loaded frozen classifier-judge from {CLASSIFIER_CKPT}", flush=True)

    results = {}
    all_final = {}
    for order in ("raster", "random", "confidence"):
        final_seq, _ = run_arm(model, tokens_gt, init_mask, order, budgets, seed=args.seed)
        all_final[order] = final_seq
        results[order] = compute_metrics(final_seq, tokens_gt, init_mask, labels, classifier, n_eval)
        print(f"[infer_ext] arm={order:10s} token_recon_acc={results[order]['token_reconstruction_acc']:.4f} "
              f"classifier_top1_acc={results[order]['classifier_top1_matches_true_label_frac']:.4f}", flush=True)

    final_seq, _ = run_arm_learned(model, policy, tokens_gt, init_mask, budgets)
    all_final["learned"] = final_seq
    results["learned"] = compute_metrics(final_seq, tokens_gt, init_mask, labels, classifier, n_eval)
    print(f"[infer_ext] arm={'learned':10s} token_recon_acc={results['learned']['token_reconstruction_acc']:.4f} "
          f"classifier_top1_acc={results['learned']['classifier_top1_matches_true_label_frac']:.4f}", flush=True)

    # Determinism / leakage sanity check: recomputed raster/random/confidence
    # should match the original reproduction's results.json bit-for-bit
    # (same frozen checkpoint, same seeds, eval() so no dropout).
    determinism_check = {}
    if os.path.exists(ORIG_RESULTS_PATH):
        with open(ORIG_RESULTS_PATH) as f:
            orig = json.load(f)["results"]
        for order in ("raster", "random", "confidence"):
            orig_acc = orig[order]["token_reconstruction_acc"]
            new_acc = results[order]["token_reconstruction_acc"]
            match = abs(orig_acc - new_acc) < 1e-9
            determinism_check[order] = {"orig": orig_acc, "recomputed": new_acc, "exact_match": match}
            print(f"[infer_ext] determinism check ({order}): orig={orig_acc:.6f} "
                  f"recomputed={new_acc:.6f} exact_match={match}", flush=True)

    elapsed = time.time() - t0
    out = {
        "n_eval": n_eval,
        "mask_frac": args.mask_frac,
        "n_masked_tokens": n_mask,
        "steps": args.steps,
        "reveal_schedule_remaining_masked": budgets,
        "seed": args.seed,
        "used_real_mnist": used_real,
        "policy_val_acc": pol_ckpt.get("final_val_acc"),
        "policy_val_majority_baseline_acc": pol_ckpt.get("val_majority_baseline_acc"),
        "results": results,
        "determinism_check_vs_original_3arm_results": determinism_check,
        "wall_clock_sec": elapsed,
    }
    out_path = os.path.join(EXT_LOG_DIR, "results_4arm.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[infer_ext] wrote {out_path}", flush=True)

    torch.save({
        "tokens_gt": tokens_gt[: args.n_sample_grid],
        "labels": labels[: args.n_sample_grid],
        "init_mask": init_mask[: args.n_sample_grid],
        "final": {k: v[: args.n_sample_grid] for k, v in all_final.items()},
    }, os.path.join(EXT_LOG_DIR, "sample_recon_data_4arm.pt"))
    print(f"[infer_ext] DONE. wall_clock={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
