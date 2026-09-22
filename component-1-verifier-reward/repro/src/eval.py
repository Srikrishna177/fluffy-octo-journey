"""Before/after eval: sample a fixed held-out batch of 64 from the base DDPM and
from the DDPO-finetuned DDPM (same seed for both, for a fair comparison), score
both with the frozen classifier, save sample grids and an eval JSON summary.
Also renders training-curve PNGs from the JSONL logs written during training.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from classifier import SmallCNN, reward_fn
from diffusion import DiffusionConfig, GaussianDiffusion
from unet import TinyUNet
from utils import set_seed

REPRO_DIR = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")
PLOTS_DIR = os.path.join(LOGS_DIR, "plots")
CURVES_DIR = os.path.join(LOGS_DIR, "curves")


def save_grid(images: torch.Tensor, path: str, title: str, nrow=8):
    imgs = ((images.clamp(-1, 1) + 1) / 2).squeeze(1).detach().cpu().numpy()  # [N,H,W] in [0,1]
    n = imgs.shape[0]
    ncol = nrow
    nrow_grid = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_grid, ncol, figsize=(ncol * 1.0, nrow_grid * 1.0))
    axes = axes.flatten()
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_loss_curve(jsonl_path, out_path, key, title, xkey="step"):
    rows = [r for r in load_jsonl(jsonl_path) if r.get("event") == "step"]
    if not rows:
        return
    xs = [r[xkey] for r in rows]
    ys = [r[key] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys)
    ax.set_xlabel(xkey)
    ax.set_ylabel(key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ddpo_reward_curve(jsonl_path, out_path):
    rows = [r for r in load_jsonl(jsonl_path) if r.get("event") == "iter"]
    if not rows:
        return
    its = [r["it"] for r in rows]
    means = [r["reward_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(its, means, label="batch mean reward")
    if len(rows) > 5:
        window = 5
        smooth = [sum(means[max(0, i - window):i + 1]) / len(means[max(0, i - window):i + 1]) for i in range(len(means))]
        ax.plot(its, smooth, label=f"rolling mean (w={window})", linewidth=2)
    ax.set_xlabel("RL iteration")
    ax.set_ylabel("mean reward (classifier confidence for target digit)")
    ax.set_title("DDPO fine-tuning: reward over RL iterations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    kl_rows = [r for r in rows if "kl_to_base" in r]
    if kl_rows:
        its_kl = [r["it"] for r in kl_rows]
        kls = [r["kl_to_base"] for r in kl_rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(its_kl, kls, marker="o")
        ax.set_xlabel("RL iteration")
        ax.set_ylabel("mean per-step KL(policy || base)")
        ax.set_title("KL-to-base-model estimate over RL fine-tuning")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "ddpo_kl_to_base.png"), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--target_class", type=int, default=7)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    diffusion = GaussianDiffusion(DiffusionConfig(T=args.T))

    classifier = SmallCNN()
    classifier.load_state_dict(torch.load(os.path.join(CKPT_DIR, "classifier.pt"), map_location="cpu"))
    classifier.eval()

    base = TinyUNet()
    base.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ddpm_base.pt"), map_location="cpu"))
    base.eval()

    finetuned = TinyUNet()
    finetuned.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ddpo_finetuned.pt"), map_location="cpu"))
    finetuned.eval()

    set_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    traj_before = diffusion.sample_trajectory(base, args.n_eval, img_size=16, generator=gen)
    images_before = traj_before["x0_final"]
    reward_before = reward_fn(classifier, images_before, args.target_class)

    gen2 = torch.Generator().manual_seed(args.seed)  # same noise seed for a fair before/after comparison
    traj_after = diffusion.sample_trajectory(finetuned, args.n_eval, img_size=16, generator=gen2)
    images_after = traj_after["x0_final"]
    reward_after = reward_fn(classifier, images_after, args.target_class)

    # also report the predicted class distribution to check for reward hacking
    with torch.no_grad():
        probs_before = torch.softmax(classifier(images_before), dim=-1)
        probs_after = torch.softmax(classifier(images_after), dim=-1)
    pred_before = probs_before.argmax(-1)
    pred_after = probs_after.argmax(-1)
    frac_target_before = (pred_before == args.target_class).float().mean().item()
    frac_target_after = (pred_after == args.target_class).float().mean().item()

    summary = {
        "n_eval": args.n_eval,
        "target_class": args.target_class,
        "seed": args.seed,
        "reward_mean_before": reward_before.mean().item(),
        "reward_std_before": reward_before.std().item(),
        "reward_mean_after": reward_after.mean().item(),
        "reward_std_after": reward_after.std().item(),
        "reward_delta": reward_after.mean().item() - reward_before.mean().item(),
        "frac_argmax_is_target_before": frac_target_before,
        "frac_argmax_is_target_after": frac_target_after,
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(LOGS_DIR, "eval_before_after.json"), "w") as f:
        json.dump(summary, f, indent=2)

    save_grid(images_before, os.path.join(PLOTS_DIR, "samples_before_rl.png"),
              f"Before DDPO RL fine-tuning (mean reward={summary['reward_mean_before']:.3f})")
    save_grid(images_after, os.path.join(PLOTS_DIR, "samples_after_rl.png"),
              f"After DDPO RL fine-tuning (mean reward={summary['reward_mean_after']:.3f})")

    plot_loss_curve(os.path.join(CURVES_DIR, "classifier_train.jsonl"),
                     os.path.join(PLOTS_DIR, "classifier_loss.png"), "loss", "Classifier pretraining loss")
    plot_loss_curve(os.path.join(CURVES_DIR, "ddpm_train.jsonl"),
                     os.path.join(PLOTS_DIR, "ddpm_loss.png"), "avg_loss_100", "DDPM pretraining loss (100-step avg)")
    plot_ddpo_reward_curve(os.path.join(CURVES_DIR, "ddpo_train.jsonl"),
                            os.path.join(PLOTS_DIR, "ddpo_reward_curve.png"))


if __name__ == "__main__":
    main()
