"""3-arm MaskGIT decoding-order sweep (reproduction, DECISION.md).

Reuses ONE frozen trained model (checkpoints/maskgit.pt) for all three arms:
  - raster: fixed left-to-right/top-to-bottom reveal order
  - random: uniformly shuffled reveal order (fixed once per image up front,
    which is distributionally identical to drawing a fresh uniform random
    subset of the still-masked positions at every step)
  - confidence: MaskGIT's own mechanism -- reveal the currently-masked
    positions the model is most confident about at each step

Same starting 75%-masked sequence, same per-step reveal budget (MaskGIT's own
cosine unmasking schedule, computed once from the confidence arm's schedule
and applied identically to all three arms) -- only the ORDER differs.

Metric: final token-reconstruction accuracy against ground truth on the
originally-masked positions. Secondary: component-1's frozen MNIST classifier
scores the dequantized reconstruction against the known true digit label.
"""
import argparse
import json
import math
import os

import torch

from data import load_mnist16_tokens, MASK_TOKEN, SEQ_LEN, dequantize_tokens
from model import BidirectionalMaskedTransformer
from classifier import SmallCNN

HERE = os.path.dirname(__file__)
CKPT_PATH = os.path.join(HERE, "..", "checkpoints", "maskgit.pt")
CLASSIFIER_CKPT = os.path.join(
    HERE, "..", "..", "..", "component-1-verifier-reward", "repro", "checkpoints", "classifier.pt"
)
LOG_DIR = os.path.join(HERE, "..", "..", "logs")


def maskgit_reveal_schedule(n_init: int, T: int):
    """counts[0]=n_init (fully masked), counts[T]=0 (fully revealed). MaskGIT's
    cosine schedule gamma(r)=cos(r*pi/2) applied to the REMAINING masked count."""
    counts = [n_init]
    for t in range(1, T + 1):
        r = t / T
        remaining = n_init * math.cos(r * math.pi / 2.0)
        remaining = int(round(remaining))
        remaining = min(remaining, counts[-1])
        remaining = max(remaining, 0)
        counts.append(remaining)
    counts[-1] = 0
    return counts


def run_arm(model, tokens_gt: torch.Tensor, init_mask: torch.Tensor, order: str,
            budgets, seed: int):
    """tokens_gt: [N,256] long. init_mask: [N,256] bool (True=masked, SAME for all arms).
    order: 'raster' | 'random' | 'confidence'.
    Returns (final_seq [N,256] long, snapshots dict{step: [N,256] long})."""
    N, L = tokens_gt.shape
    T = len(budgets) - 1
    seq = tokens_gt.clone()
    seq[init_mask] = MASK_TOKEN
    still_masked = init_mask.clone()

    n_init_per_img = init_mask.sum(dim=1)
    assert (n_init_per_img == n_init_per_img[0]).all(), "expected equal initial mask count per image"
    n_init = int(n_init_per_img[0].item())

    order_idx = None
    if order in ("raster", "random"):
        g = torch.Generator().manual_seed(seed)
        rows = []
        for i in range(N):
            idx = torch.nonzero(init_mask[i], as_tuple=True)[0]  # ascending position index
            if order == "random":
                perm = torch.randperm(idx.shape[0], generator=g)
                idx = idx[perm]
            rows.append(idx)
        order_idx = torch.stack(rows, dim=0)  # [N, n_init]

    cum_revealed = 0
    snapshots = {0: seq.clone()}
    with torch.no_grad():
        for t in range(1, T + 1):
            logits = model(seq)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)  # [N,L]
            n_reveal = budgets[t - 1] - budgets[t]
            if n_reveal > 0:
                if order in ("raster", "random"):
                    sel = order_idx[:, cum_revealed: cum_revealed + n_reveal]
                    pred_at_sel = torch.gather(pred, 1, sel)
                    seq.scatter_(1, sel, pred_at_sel)
                    still_masked.scatter_(1, sel, False)
                else:  # confidence
                    conf_masked = conf.masked_fill(~still_masked, -1.0)
                    _, topk_idx = torch.topk(conf_masked, n_reveal, dim=1)
                    pred_at_sel = torch.gather(pred, 1, topk_idx)
                    seq.scatter_(1, topk_idx, pred_at_sel)
                    still_masked.scatter_(1, topk_idx, False)
                cum_revealed += n_reveal
            snapshots[t] = seq.clone()
    assert not still_masked.any(), "all masked positions should be revealed by the last step"
    return seq, snapshots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--n_test_pool", type=int, default=2000)
    ap.add_argument("--mask_frac", type=float, default=0.75)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n_sample_grid", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    torch.manual_seed(args.seed)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = BidirectionalMaskedTransformer()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer] loaded model from {CKPT_PATH} (trained step={ckpt.get('step')}, "
          f"n_params={ckpt.get('n_params')})", flush=True)

    _, _, test_tokens, test_labels, used_real = load_mnist16_tokens(
        n_train=2, n_test=args.n_test_pool, seed=0
    )
    n_eval = min(args.n_eval, test_tokens.shape[0])
    tokens_gt = test_tokens[:n_eval]  # [n_eval, 256]
    labels = test_labels[:n_eval]
    print(f"[infer] eval set: {n_eval} held-out MNIST test images, used_real_mnist={used_real}", flush=True)

    # SAME initial random mask for every arm (fixed 75% of 256 positions per image)
    n_mask = int(round(args.mask_frac * SEQ_LEN))
    g = torch.Generator().manual_seed(args.seed)
    rand_scores = torch.rand(n_eval, SEQ_LEN, generator=g)
    sorted_scores, _ = torch.sort(rand_scores, dim=1)
    thresh = sorted_scores[:, n_mask - 1: n_mask]
    init_mask = rand_scores <= thresh  # bool [n_eval, 256], exactly n_mask True per row
    assert (init_mask.sum(dim=1) == n_mask).all()

    budgets = maskgit_reveal_schedule(n_mask, args.steps)
    print(f"[infer] mask_frac={args.mask_frac} -> {n_mask}/{SEQ_LEN} masked tokens; "
          f"steps={args.steps}; reveal-budget schedule (remaining masked count per step): {budgets}", flush=True)

    classifier = SmallCNN(num_classes=10)
    cls_ckpt = torch.load(CLASSIFIER_CKPT, map_location="cpu")
    state = cls_ckpt["model"] if isinstance(cls_ckpt, dict) and "model" in cls_ckpt else cls_ckpt
    classifier.load_state_dict(state)
    classifier.eval()
    print(f"[infer] loaded frozen classifier-judge from {CLASSIFIER_CKPT}", flush=True)

    results = {}
    all_final = {}
    all_snapshots = {}
    for order in ("raster", "random", "confidence"):
        final_seq, snapshots = run_arm(model, tokens_gt, init_mask, order, budgets, seed=args.seed)
        all_final[order] = final_seq
        all_snapshots[order] = snapshots

        correct = (final_seq == tokens_gt) & init_mask
        token_acc = correct.sum().item() / init_mask.sum().item()

        recon_img = dequantize_tokens(final_seq).view(n_eval, 1, 16, 16)
        with torch.no_grad():
            logits = classifier(recon_img)
            probs = torch.softmax(logits, dim=-1)
        true_class_conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pred_class = probs.argmax(dim=1)
        classifier_acc = (pred_class == labels).float().mean().item()
        classifier_true_conf_mean = true_class_conf.mean().item()

        # per-image std for a rough noise estimate
        per_image_acc = correct.float().sum(dim=1) / init_mask.float().sum(dim=1)
        results[order] = {
            "token_reconstruction_acc": token_acc,
            "token_acc_std_across_images": per_image_acc.std().item(),
            "classifier_true_label_confidence_mean": classifier_true_conf_mean,
            "classifier_top1_matches_true_label_frac": classifier_acc,
        }
        print(f"[infer] arm={order:10s} token_recon_acc={token_acc:.4f} "
              f"(std across images={per_image_acc.std().item():.4f}) "
              f"classifier_true_conf={classifier_true_conf_mean:.4f} "
              f"classifier_top1_acc={classifier_acc:.4f}", flush=True)

    out = {
        "n_eval": n_eval,
        "mask_frac": args.mask_frac,
        "n_masked_tokens": n_mask,
        "steps": args.steps,
        "reveal_schedule_remaining_masked": budgets,
        "seed": args.seed,
        "used_real_mnist": used_real,
        "results": results,
    }
    out_path = os.path.join(LOG_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[infer] wrote {out_path}", flush=True)

    # save raw tensors needed for plotting (ground truth, init mask, final reconstructions)
    torch.save({
        "tokens_gt": tokens_gt[: args.n_sample_grid],
        "labels": labels[: args.n_sample_grid],
        "init_mask": init_mask[: args.n_sample_grid],
        "final": {k: v[: args.n_sample_grid] for k, v in all_final.items()},
    }, os.path.join(LOG_DIR, "sample_recon_data.pt"))
    print("[infer] DONE", flush=True)


if __name__ == "__main__":
    main()
