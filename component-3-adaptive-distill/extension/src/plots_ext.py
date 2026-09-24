"""Render the extension's plots from its own logged JSON/JSONL/checkpoint
artifacts. Read-only; no training happens here.

Produces:
  - onpolicy_training_curve.png: this run's on-policy round-1 loss curve,
    overlaid with the reproduction's own round-1 (data-marginal) loss curve
    for direct comparison (same schedule transition, same budget).
  - sample_grid_comparison.png: teacher (32-step) vs. data-marginal 16-step
    student vs. on-policy 16-step student, same starting noise -- same visual
    format as repro/logs/plots/sample_grid_by_nfe.png.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

EXT_DIR = os.path.join(os.path.dirname(__file__), "..")
REPRO_DIR = os.path.join(EXT_DIR, "..", "repro")
REPRO_LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
LOGS_DIR = os.path.join(EXT_DIR, "logs")
CKPT_DIR = os.path.join(EXT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(LOGS_DIR, "plots")
CURVES_DIR = os.path.join(LOGS_DIR, "curves")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_training_curve():
    onpolicy_rows = [r for r in load_jsonl(os.path.join(CURVES_DIR, "distill_train_onpolicy.jsonl"))
                      if r.get("event") == "step"]
    repro_curve_path = os.path.join(REPRO_LOGS_DIR, "curves", "distill_train.jsonl")
    fig, ax = plt.subplots(figsize=(7.5, 5))

    xs = [r["step"] for r in onpolicy_rows]
    ys = [r["avg_loss_50"] for r in onpolicy_rows]
    ax.plot(xs, ys, label="on-policy (this extension), round 1: 32→16", color="tab:orange")

    if os.path.exists(repro_curve_path):
        repro_rows = [r for r in load_jsonl(repro_curve_path)
                      if r.get("event") == "step" and r.get("round") == 1]
        if repro_rows:
            xs2 = [r["step"] for r in repro_rows]
            ys2 = [r["avg_loss_50"] for r in repro_rows]
            ax.plot(xs2, ys2, label="data-marginal (reproduction), round 1: 32→16", color="tab:blue")

    ax.set_xlabel("training step (within round, same lr/batch_size/budget)")
    ax.set_ylabel("distillation loss (50-step rolling avg, MSE on implied x0)")
    ax.set_title("Round 1 (32→16) training loss: on-policy vs. data-marginal states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "onpolicy_training_curve.png"), dpi=130)
    plt.close(fig)


def plot_sample_grid():
    samples = torch.load(os.path.join(CKPT_DIR, "eval_samples.pt"), map_location="cpu")
    rows = [
        ("teacher_32", "teacher\n(32 NFE)"),
        ("student_16_data_marginal", "data-marginal\nstudent (16 NFE)"),
        ("student_16_onpolicy", "on-policy\nstudent (16 NFE)"),
    ]
    n_show = 8
    fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 1.1, len(rows) * 1.4))
    for r, (key, label) in enumerate(rows):
        imgs = ((samples[key][:n_show].clamp(-1, 1) + 1) / 2).squeeze(1).numpy()
        for col in range(n_show):
            ax = axes[r, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("Same starting noise (seed 12345): teacher vs. data-marginal vs. on-policy 16-step student",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "sample_grid_comparison.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_training_curve()
    plot_sample_grid()
    print("[plots_ext] wrote onpolicy_training_curve.png, sample_grid_comparison.png")


if __name__ == "__main__":
    main()
