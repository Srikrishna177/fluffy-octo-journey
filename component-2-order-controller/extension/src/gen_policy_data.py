"""Generate the (feature, is_correct) supervised dataset for policy training.

Per extension/DECISION.md:
  - Uses the FROZEN base transformer (repro/checkpoints/maskgit.pt), weights
    untouched, eval() mode, no_grad throughout.
  - Uses the MNIST TRAIN split (the identical 20,000-image pool the base
    transformer trained on -- reproduced deterministically by calling
    load_mnist16_tokens(n_train=20000, n_test=2, seed=0), the same
    (n_train, seed) the base transformer's train_maskgit.py used; n_test does
    not affect which images land in train_tokens, see RUN_LOG.md leakage note).
  - For each image, samples ONE random 75%-mask (same mask_frac as the
    reproduction's eval condition, since that is the regime the policy will
    be used in), runs the frozen model once, and for every MASKED position
    records: feature vector (policy_common.extract_features) and binary label
    (argmax prediction == true token).
  - This uses the TRAIN split only. The eval-pool (MNIST test split) used by
    the reproduction's 3-arm comparison is never touched here -- see
    inference_ext.py / RUN_LOG.md for the explicit leakage check.
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(__file__)
REPRO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "repro", "src"))
sys.path.insert(0, REPRO_SRC)
sys.path.insert(0, HERE)

from data import load_mnist16_tokens, MASK_TOKEN, SEQ_LEN  # noqa: E402
from model import BidirectionalMaskedTransformer  # noqa: E402
from policy_common import forward_with_hidden, extract_features, FEATURE_DIM, FEATURE_SPEC  # noqa: E402

CKPT_PATH = os.path.join(HERE, "..", "..", "repro", "checkpoints", "maskgit.pt")
DATA_DIR = os.path.join(HERE, "..", "data")


def sample_fixed_frac_mask(n: int, seq_len: int, mask_frac: float, generator) -> torch.Tensor:
    """Exactly round(mask_frac*seq_len) masked positions per row, uniformly random."""
    n_mask = int(round(mask_frac * seq_len))
    rand_scores = torch.rand(n, seq_len, generator=generator)
    sorted_scores, _ = torch.sort(rand_scores, dim=1)
    thresh = sorted_scores[:, n_mask - 1: n_mask]
    mask = rand_scores <= thresh
    assert (mask.sum(dim=1) == n_mask).all()
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train_images", type=int, default=20000,
                     help="size of the MNIST train pool to draw masked-position "
                          "supervision from (same pool the base transformer trained on)")
    ap.add_argument("--mask_frac", type=float, default=0.75)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=999,
                     help="separate seed for the policy-data masking draw (distinct from "
                          "the base-training seed=0 and the eval seed=12345)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = BidirectionalMaskedTransformer()
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[gen_policy_data] loaded frozen model from {CKPT_PATH} "
          f"(trained step={ckpt.get('step')})", flush=True)

    # Identical (n_train, seed) as train_maskgit.py's call -> identical 20k train pool.
    # n_test is irrelevant to which images land in train_tokens (see RUN_LOG.md).
    train_tokens, train_labels, _, _, used_real = load_mnist16_tokens(
        n_train=args.n_train_images, n_test=2, seed=0
    )
    print(f"[gen_policy_data] loaded {train_tokens.shape[0]} MNIST TRAIN-split images "
          f"(used_real_mnist={used_real})", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    full_mask = sample_fixed_frac_mask(train_tokens.shape[0], SEQ_LEN, args.mask_frac, g)

    all_X, all_y = [], []
    n = train_tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch_gt = train_tokens[start:end]
            batch_mask = full_mask[start:end]
            inp = batch_gt.clone()
            inp[batch_mask] = MASK_TOKEN

            hidden, logits = forward_with_hidden(model, inp)
            feat, pred = extract_features(hidden, logits, SEQ_LEN)  # [b,L,99], [b,L]
            correct = (pred == batch_gt).float()  # [b,L]

            feat_flat = feat[batch_mask]  # [n_masked_in_batch, 99]
            label_flat = correct[batch_mask]  # [n_masked_in_batch]
            all_X.append(feat_flat)
            all_y.append(label_flat)
            if (start // args.batch_size) % 10 == 0:
                print(f"[gen_policy_data] processed {end}/{n} images", flush=True)

    X = torch.cat(all_X, dim=0)
    y = torch.cat(all_y, dim=0)
    pos_rate = y.mean().item()
    elapsed = time.time() - t0
    print(f"[gen_policy_data] DONE. X={tuple(X.shape)} y={tuple(y.shape)} "
          f"positive_rate(base_model_argmax_correct)={pos_rate:.4f} "
          f"elapsed={elapsed:.1f}s", flush=True)

    out_path = os.path.join(DATA_DIR, "policy_dataset.pt")
    torch.save({
        "X": X, "y": y,
        "n_images": n,
        "mask_frac": args.mask_frac,
        "data_seed": args.seed,
        "feature_dim": FEATURE_DIM,
        "feature_spec": FEATURE_SPEC,
        "positive_rate": pos_rate,
        "elapsed_sec": elapsed,
        "source_split": "MNIST train (torchvision train=True), NOT the eval test-pool",
    }, out_path)
    print(f"[gen_policy_data] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
