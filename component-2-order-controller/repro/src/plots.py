"""Plots: training curve, results bar chart, and per-arm sample reconstruction grids."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import dequantize_tokens

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "..", "..", "logs")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
CURVE_DIR = os.path.join(LOG_DIR, "curves")


def plot_training_curve():
    path = os.path.join(CURVE_DIR, "maskgit_train.jsonl")
    steps, losses, avg_losses, accs = [], [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            steps.append(rec["step"])
            losses.append(rec["loss"])
            avg_losses.append(rec["avg_loss_100"])
            accs.append(rec["masked_token_acc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(steps, losses, alpha=0.3, label="loss (per logged step)", color="#4C78A8")
    ax1.plot(steps, avg_losses, label="loss (100-step rolling avg)", color="#E45756", linewidth=2)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("masked-token cross-entropy loss")
    ax1.set_title("MaskGIT pretraining loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, accs, color="#54A24B")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("masked-token accuracy")
    ax2.set_title("Masked-token accuracy (train batches)")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "maskgit_training_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots] wrote {out}")


def plot_results_bar():
    with open(os.path.join(LOG_DIR, "results.json")) as f:
        results = json.load(f)["results"]
    arms = ["raster", "random", "confidence"]
    accs = [results[a]["token_reconstruction_acc"] for a in arms]
    cls = [results[a]["classifier_true_label_confidence_mean"] for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    colors = ["#9D9D9D", "#B279A2", "#4C78A8"]
    ax1.bar(arms, accs, color=colors)
    ax1.set_ylabel("token reconstruction accuracy")
    ax1.set_title("Final token-recon accuracy by reveal order")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center")

    ax2.bar(arms, cls, color=colors)
    ax2.set_ylabel("classifier confidence in true digit")
    ax2.set_title("Classifier-judge score by reveal order")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(cls):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "arms_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots] wrote {out}")


def plot_sample_grids():
    data = torch.load(os.path.join(LOG_DIR, "sample_recon_data.pt"), map_location="cpu")
    tokens_gt = data["tokens_gt"]      # [n,256]
    init_mask = data["init_mask"]      # [n,256]
    finals = data["final"]             # dict order -> [n,256]
    n = tokens_gt.shape[0]

    masked_view = tokens_gt.clone()
    masked_view[init_mask] = -1  # sentinel for "still masked" display

    def to_img(tok_row, masked_sentinel=False):
        if masked_sentinel:
            img = tok_row.clone().float()
            mask = img < 0
            img = dequantize_tokens(img.clamp(min=0))
            img[mask] = float("nan")  # show as blank/gray via colormap
            return img.view(16, 16).numpy()
        return dequantize_tokens(tok_row).view(16, 16).numpy()

    arms = ["raster", "random", "confidence"]
    ncols = 2 + len(arms)
    fig, axes = plt.subplots(n, ncols, figsize=(2.0 * ncols, 2.0 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["ground truth", "75% masked input"] + [f"{a} order" for a in arms]
    for r in range(n):
        gt_img = to_img(tokens_gt[r])
        masked_img = to_img(masked_view[r], masked_sentinel=True)
        masked_cmap = plt.get_cmap("gray").copy()
        masked_cmap.set_bad(color="#D9534F")  # red = still-masked position
        axes[r, 0].imshow(gt_img, cmap="gray", vmin=-1, vmax=1)
        axes[r, 1].imshow(masked_img, cmap=masked_cmap, vmin=-1, vmax=1)
        for c, arm in enumerate(arms):
            recon_img = to_img(finals[arm][r])
            axes[r, 2 + c].imshow(recon_img, cmap="gray", vmin=-1, vmax=1)
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=10)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "sample_reconstructions.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plots] wrote {out}")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_training_curve()
    plot_results_bar()
    plot_sample_grids()
