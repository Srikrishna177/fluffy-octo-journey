"""Extension plots: policy training curve, 4-arm results bar chart, and a
4-column (raster/random/confidence/learned) sample reconstruction grid.
Style matches repro/src/plots.py, extended by one arm/column.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(__file__)
REPRO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "repro", "src"))
sys.path.insert(0, REPRO_SRC)
from data import dequantize_tokens  # noqa: E402

EXT_DIR = os.path.abspath(os.path.join(HERE, ".."))
EXT_LOG_DIR = os.path.join(EXT_DIR, "logs")
PLOT_DIR = os.path.join(EXT_LOG_DIR, "plots")
CURVE_DIR = os.path.join(EXT_LOG_DIR, "curves")

ARMS = ["raster", "random", "confidence", "learned"]
COLORS = ["#9D9D9D", "#B279A2", "#4C78A8", "#E45756"]


def plot_policy_training_curve():
    path = os.path.join(CURVE_DIR, "policy_train.jsonl")
    epochs, train_loss, val_loss, train_acc, val_acc, maj = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            epochs.append(rec["epoch"])
            train_loss.append(rec["train_loss"])
            val_loss.append(rec["val_loss"])
            train_acc.append(rec["train_acc"])
            val_acc.append(rec["val_acc"])
            maj.append(rec["val_majority_baseline_acc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_loss, label="train loss", color="#4C78A8", marker="o")
    ax1.plot(epochs, val_loss, label="val loss", color="#E45756", marker="o")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Policy MLP training loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, label="train acc", color="#54A24B", marker="o")
    ax2.plot(epochs, val_acc, label="val acc", color="#F58518", marker="o")
    ax2.plot(epochs, maj, label="val majority-class baseline", color="#9D9D9D", linestyle="--")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (predict is-correct)")
    ax2.set_title("Policy MLP accuracy vs. majority baseline")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "policy_training_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots_ext] wrote {out}")


def plot_results_bar_4arm():
    with open(os.path.join(EXT_LOG_DIR, "results_4arm.json")) as f:
        results = json.load(f)["results"]
    accs = [results[a]["token_reconstruction_acc"] for a in ARMS]
    cls = [results[a]["classifier_true_label_confidence_mean"] for a in ARMS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(ARMS, accs, color=COLORS)
    ax1.set_ylabel("token reconstruction accuracy")
    ax1.set_title("Final token-recon accuracy by reveal order (4 arms)")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center")

    ax2.bar(ARMS, cls, color=COLORS)
    ax2.set_ylabel("classifier confidence in true digit")
    ax2.set_title("Classifier-judge score by reveal order (4 arms)")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(cls):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "arms_comparison_4arm.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots_ext] wrote {out}")


def plot_sample_grids_4arm():
    data = torch.load(os.path.join(EXT_LOG_DIR, "sample_recon_data_4arm.pt"), map_location="cpu")
    tokens_gt = data["tokens_gt"]
    init_mask = data["init_mask"]
    finals = data["final"]
    n = tokens_gt.shape[0]

    masked_view = tokens_gt.clone()
    masked_view[init_mask] = -1

    def to_img(tok_row, masked_sentinel=False):
        if masked_sentinel:
            img = tok_row.clone().float()
            mask = img < 0
            img = dequantize_tokens(img.clamp(min=0))
            img[mask] = float("nan")
            return img.view(16, 16).numpy()
        return dequantize_tokens(tok_row).view(16, 16).numpy()

    ncols = 2 + len(ARMS)
    fig, axes = plt.subplots(n, ncols, figsize=(2.0 * ncols, 2.0 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["ground truth", "75% masked input"] + [f"{a} order" for a in ARMS]
    for r in range(n):
        gt_img = to_img(tokens_gt[r])
        masked_img = to_img(masked_view[r], masked_sentinel=True)
        masked_cmap = plt.get_cmap("gray").copy()
        masked_cmap.set_bad(color="#D9534F")
        axes[r, 0].imshow(gt_img, cmap="gray", vmin=-1, vmax=1)
        axes[r, 1].imshow(masked_img, cmap=masked_cmap, vmin=-1, vmax=1)
        for c, arm in enumerate(ARMS):
            recon_img = to_img(finals[arm][r])
            axes[r, 2 + c].imshow(recon_img, cmap="gray", vmin=-1, vmax=1)
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=10)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "sample_reconstructions_4arm.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots_ext] wrote {out}")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_policy_training_curve()
    plot_results_bar_4arm()
    plot_sample_grids_4arm()
