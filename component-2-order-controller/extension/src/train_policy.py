"""Train the small MLP policy on the (feature, is_correct) dataset produced by
gen_policy_data.py. Purely a downstream supervised classifier -- does not
touch the frozen MaskGIT transformer's weights at all.

Train/val split here is an internal 90/10 split of the POLICY dataset itself
(both drawn from the MNIST train pool) -- used only to monitor the policy's
own fit/overfit, and is unrelated to the separate, much bigger leakage
guarantee that the eval-pool (MNIST test split) is never in this dataset at
all (see gen_policy_data.py / RUN_LOG.md).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from policy_common import PolicyMLP, FEATURE_DIM  # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "data")
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")
CURVE_DIR = os.path.join(HERE, "..", "logs", "curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)
    torch.manual_seed(args.seed)

    blob = torch.load(os.path.join(DATA_DIR, "policy_dataset.pt"), map_location="cpu")
    X, y = blob["X"], blob["y"]
    n = X.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"[train_policy] dataset n={n} (feature_dim={blob['feature_dim']}, "
          f"positive_rate={blob['positive_rate']:.4f}); train={X_train.shape[0]} "
          f"val={X_val.shape[0]}", flush=True)

    model = PolicyMLP(in_dim=FEATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    curve_path = os.path.join(CURVE_DIR, "policy_train.jsonl")
    curve_f = open(curve_path, "w")

    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logit = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logit) > 0.5).float()
                ep_correct += (pred == yb).sum().item()
                ep_n += yb.shape[0]
                ep_loss += loss.item() * yb.shape[0]
            global_step += 1
        train_loss = ep_loss / ep_n
        train_acc = ep_correct / ep_n

        model.eval()
        with torch.no_grad():
            val_logit = model(X_val)
            val_loss = loss_fn(val_logit, y_val).item()
            val_pred = (torch.sigmoid(val_logit) > 0.5).float()
            val_acc = (val_pred == y_val).float().mean().item()
            # baseline: always predict the majority class
            majority_acc = max(y_val.mean().item(), 1 - y_val.mean().item())

        elapsed = time.time() - t0
        rec = {"epoch": epoch, "step": global_step, "elapsed_sec": round(elapsed, 2),
               "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc, "val_majority_baseline_acc": majority_acc}
        curve_f.write(json.dumps(rec) + "\n")
        curve_f.flush()
        print(f"[train_policy] epoch={epoch} elapsed={elapsed:.1f}s train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"(majority_baseline={majority_acc:.4f})", flush=True)

    curve_f.close()
    total_elapsed = time.time() - t0

    torch.save({
        "model": model.state_dict(),
        "feature_dim": FEATURE_DIM,
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "val_majority_baseline_acc": majority_acc,
        "epochs": args.epochs,
        "total_elapsed_sec": total_elapsed,
    }, os.path.join(CKPT_DIR, "policy.pt"))
    print(f"[train_policy] DONE. total_elapsed={total_elapsed:.1f}s "
          f"final_val_acc={val_acc:.4f} (majority_baseline={majority_acc:.4f}) "
          f"checkpoint=extension/checkpoints/policy.pt", flush=True)


if __name__ == "__main__":
    main()
