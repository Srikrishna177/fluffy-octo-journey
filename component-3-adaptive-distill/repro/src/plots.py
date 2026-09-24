"""Render training-curve and sample-grid plots from the distillation run's
logged JSONL/JSON/checkpoint artifacts. Reads only; no training happens here."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))

REPRO_DIR = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")
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


def plot_round_curves():
    rows = [r for r in load_jsonl(os.path.join(CURVES_DIR, "distill_train.jsonl")) if r.get("event") == "step"]
    rounds = sorted(set(r["round"] for r in rows))
    labels = {1: "round 1: 32→16", 2: "round 2: 16→8", 3: "round 3: 8→4", 4: "round 4: 4→2", 5: "round 5: 2→1"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for rd in rounds:
        rr = [r for r in rows if r["round"] == rd]
        xs = [r["step"] for r in rr]
        ys = [r["avg_loss_50"] for r in rr]
        ax.plot(xs, ys, label=labels.get(rd, f"round {rd}"))
    ax.set_xlabel("training step (within round)")
    ax.set_ylabel("distillation loss (50-step rolling avg, MSE on implied x0)")
    ax.set_title("Progressive distillation: per-round training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "distill_training_curves.png"), dpi=130)
    plt.close(fig)


def plot_metrics_vs_nfe(results):
    nfes = sorted(int(k) for k in results.keys())
    mse = [results[str(n)]["trajectory_pixel_mse_vs_teacher32"] for n in nfes]
    kl = [results[str(n)]["trajectory_classifier_kl_vs_teacher32"] for n in nfes]
    conf = [results[str(n)]["own_quality_mean_top1_conf"] for n in nfes]
    ent = [results[str(n)]["own_quality_mean_entropy"] for n in nfes]
    wall = [results[str(n)]["wall_clock_sec_per_sample"] * 1000 for n in nfes]
    pop_ent = [results[str(n)].get("population_class_entropy_nats") for n in nfes]
    maj_frac = [results[str(n)].get("frac_samples_in_majority_class") for n in nfes]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))

    ax = axes[0, 0]
    ax.plot(nfes, mse, marker="o", color="tab:red", label="pixel MSE vs teacher-32")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE (student steps)")
    ax.set_ylabel("pixel MSE vs teacher's 32-step trajectory")
    ax.set_title("Trajectory-matching fidelity vs. compute")
    ax.invert_xaxis()

    ax2 = axes[0, 1]
    ax2.plot(nfes, kl, marker="s", color="tab:purple")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("NFE (student steps)")
    ax2.set_ylabel("classifier-softmax KL(teacher || arm)")
    ax2.set_title("Secondary trajectory-fidelity check (classifier space)")
    ax2.invert_xaxis()

    ax3 = axes[1, 0]
    ax3.plot(nfes, conf, marker="o", color="tab:green", label="mean top-1 confidence")
    ax3.plot(nfes, ent, marker="^", color="tab:olive", label="mean entropy (nats)")
    ax3.set_xscale("log", base=2)
    ax3.set_xlabel("NFE (student steps)")
    ax3.set_title("Absolute quality anchor (classifier-judge on own samples)")
    ax3.invert_xaxis()
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.plot(nfes, wall, marker="o", color="tab:blue")
    ax4.set_xscale("log", base=2)
    ax4.set_yscale("log")
    ax4.set_xlabel("NFE (student steps)")
    ax4.set_ylabel("wall-clock ms / sample (this CPU sandbox)")
    ax4.set_title("Compute/latency axis")
    ax4.invert_xaxis()

    ax5 = axes[0, 2]
    ax5.plot(nfes, pop_ent, marker="o", color="tab:brown")
    ax5.axhline(2.303, color="gray", linestyle=":", linewidth=1, label="max (uniform over 10 classes)")
    ax5.set_xscale("log", base=2)
    ax5.set_xlabel("NFE (student steps)")
    ax5.set_ylabel("entropy of predicted-class histogram (nats)")
    ax5.set_title("Population diversity (mode-collapse check)")
    ax5.invert_xaxis()
    ax5.legend(fontsize=8)

    ax6 = axes[1, 2]
    ax6.plot(nfes, maj_frac, marker="o", color="tab:pink")
    ax6.set_xscale("log", base=2)
    ax6.set_ylim(0, 1.05)
    ax6.set_xlabel("NFE (student steps)")
    ax6.set_ylabel("fraction of samples in single most-common class")
    ax6.set_title("Majority-class concentration")
    ax6.invert_xaxis()

    fig.suptitle("Progressive distillation cascade: quality vs. NFE (MNIST-16x16 toy DDPM)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metrics_vs_nfe.png"), dpi=130)
    plt.close(fig)


def plot_sample_grid():
    samples = torch.load(os.path.join(CKPT_DIR, "eval_samples_by_nfe.pt"), map_location="cpu")
    nfes = [32, 16, 8, 4, 2, 1]
    n_show = 8
    fig, axes = plt.subplots(len(nfes), n_show, figsize=(n_show * 1.1, len(nfes) * 1.25))
    for row, nfe in enumerate(nfes):
        imgs = ((samples[nfe][:n_show].clamp(-1, 1) + 1) / 2).squeeze(1).numpy()
        for col in range(n_show):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                label = f"teacher\n(32 NFE)" if nfe == 32 else f"student\n({nfe} NFE)"
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("Same starting noise, teacher (32-step DDIM) vs. each distilled arm", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "sample_grid_by_nfe.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, "results.json")) as f:
        d = json.load(f)
    plot_round_curves()
    plot_metrics_vs_nfe(d["results_by_nfe"])
    plot_sample_grid()
    print("[plots] wrote distill_training_curves.png, metrics_vs_nfe.png, sample_grid_by_nfe.png")


if __name__ == "__main__":
    main()
