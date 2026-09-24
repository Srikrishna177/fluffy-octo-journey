"""Independent-judge eval for the extension's treatment run.

Protocol matches repro/src/eval.py exactly (same n_eval=64, same seed=12345,
same before/after sampling-seed trick): sample a fixed held-out batch from
the base DDPM (before) and from the internal-reward DDPO-finetuned DDPM
(after), then score BOTH with the ORIGINAL FROZEN EXTERNAL classifier
(repro/checkpoints/classifier.pt) — never the internal probe — so the
treatment's result is judged on the same yardstick as the reproduction's own
eval_before_after.json. We also report the internal-probe's own opinion of
the after-RL samples, for transparency, but the external-classifier numbers
are the ones that are directly comparable to the reproduction.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

EXT_DIR = os.path.dirname(__file__)
REPRO_SRC = os.path.join(EXT_DIR, "..", "..", "repro", "src")
sys.path.insert(0, REPRO_SRC)
sys.path.insert(0, EXT_DIR)

from classifier import SmallCNN, reward_fn  # noqa: E402
from diffusion import DiffusionConfig, GaussianDiffusion  # noqa: E402
from unet import TinyUNet  # noqa: E402
from utils import set_seed  # noqa: E402
from probe import FeatureExtractor, LinearProbe, internal_reward_fn  # noqa: E402

EXT_ROOT = os.path.join(EXT_DIR, "..")
LOGS_DIR = os.path.join(EXT_ROOT, "logs")
CKPT_DIR = os.path.join(EXT_ROOT, "checkpoints")
PLOTS_DIR = os.path.join(LOGS_DIR, "plots")
CURVES_DIR = os.path.join(LOGS_DIR, "curves")
REPRO_CKPT_DIR = os.path.join(EXT_DIR, "..", "..", "repro", "checkpoints")


def save_grid(images: torch.Tensor, path: str, title: str, nrow=8):
    imgs = ((images.clamp(-1, 1) + 1) / 2).squeeze(1).detach().cpu().numpy()
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


def plot_probe_loss(jsonl_path, out_path):
    rows = [r for r in load_jsonl(jsonl_path) if r.get("event") == "step"]
    if not rows:
        return
    xs = [r["step"] for r in rows]
    ys = [r["loss"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Internal-probe training loss (linear head on frozen DDPM features)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ddpo_reward_and_kl(jsonl_path, out_reward_path, out_kl_path, title_suffix):
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
    ax.set_ylabel("mean reward (internal-probe confidence for target digit)")
    ax.set_title(f"DDPO fine-tuning: reward over RL iterations ({title_suffix})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_reward_path, dpi=120)
    plt.close(fig)

    kl_rows = [r for r in rows if "kl_to_base" in r]
    if kl_rows:
        its_kl = [r["it"] for r in kl_rows]
        kls = [r["kl_to_base"] for r in kl_rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(its_kl, kls, marker="o")
        ax.set_xlabel("RL iteration")
        ax.set_ylabel("mean per-step KL(policy || base)")
        ax.set_title(f"KL-to-base-model estimate over RL fine-tuning ({title_suffix})")
        fig.tight_layout()
        fig.savefig(out_kl_path, dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--target_class", type=int, default=7)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--base_ckpt", type=str, default=os.path.join(REPRO_CKPT_DIR, "ddpm_base.pt"))
    ap.add_argument("--finetuned_ckpt", type=str, default=os.path.join(CKPT_DIR, "ddpo_internal_finetuned.pt"))
    ap.add_argument("--external_classifier_ckpt", type=str, default=os.path.join(REPRO_CKPT_DIR, "classifier.pt"))
    ap.add_argument("--probe_ckpt", type=str, default=os.path.join(CKPT_DIR, "probe.pt"))
    args = ap.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    diffusion = GaussianDiffusion(DiffusionConfig(T=args.T))

    external_classifier = SmallCNN()
    external_classifier.load_state_dict(torch.load(args.external_classifier_ckpt, map_location="cpu"))
    external_classifier.eval()

    base = TinyUNet()
    base.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    base.eval()

    finetuned = TinyUNet()
    finetuned.load_state_dict(torch.load(args.finetuned_ckpt, map_location="cpu"))
    finetuned.eval()

    # internal probe, for transparency (NOT the judge metric)
    feat_backbone = TinyUNet()
    feat_backbone.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    extractor = FeatureExtractor(feat_backbone)
    probe = LinearProbe()
    probe.load_state_dict(torch.load(args.probe_ckpt, map_location="cpu"))
    probe.eval()

    set_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    traj_before = diffusion.sample_trajectory(base, args.n_eval, img_size=16, generator=gen)
    images_before = traj_before["x0_final"]

    gen2 = torch.Generator().manual_seed(args.seed)  # same noise seed, fair before/after comparison
    traj_after = diffusion.sample_trajectory(finetuned, args.n_eval, img_size=16, generator=gen2)
    images_after = traj_after["x0_final"]

    # ---- INDEPENDENT JUDGE: external frozen classifier (the number comparable to repro) ----
    ext_reward_before = reward_fn(external_classifier, images_before, args.target_class)
    ext_reward_after = reward_fn(external_classifier, images_after, args.target_class)
    with torch.no_grad():
        ext_probs_before = torch.softmax(external_classifier(images_before), dim=-1)
        ext_probs_after = torch.softmax(external_classifier(images_after), dim=-1)
    ext_pred_before = ext_probs_before.argmax(-1)
    ext_pred_after = ext_probs_after.argmax(-1)
    ext_frac_target_before = (ext_pred_before == args.target_class).float().mean().item()
    ext_frac_target_after = (ext_pred_after == args.target_class).float().mean().item()

    # ---- internal probe's own opinion, for transparency only ----
    int_reward_before = internal_reward_fn(extractor, probe, images_before, args.target_class)
    int_reward_after = internal_reward_fn(extractor, probe, images_after, args.target_class)

    summary = {
        "reward_source_for_training": "internal_probe",
        "judge_for_this_eval": "external_frozen_classifier (repro/checkpoints/classifier.pt)",
        "n_eval": args.n_eval,
        "target_class": args.target_class,
        "seed": args.seed,
        "external_judge_reward_mean_before": ext_reward_before.mean().item(),
        "external_judge_reward_std_before": ext_reward_before.std().item(),
        "external_judge_reward_mean_after": ext_reward_after.mean().item(),
        "external_judge_reward_std_after": ext_reward_after.std().item(),
        "external_judge_reward_delta": ext_reward_after.mean().item() - ext_reward_before.mean().item(),
        "external_judge_frac_argmax_is_target_before": ext_frac_target_before,
        "external_judge_frac_argmax_is_target_after": ext_frac_target_after,
        "internal_probe_reward_mean_before": int_reward_before.mean().item(),
        "internal_probe_reward_mean_after": int_reward_after.mean().item(),
        "internal_probe_reward_std_after": int_reward_after.std().item(),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(LOGS_DIR, "eval_external_judge.json"), "w") as f:
        json.dump(summary, f, indent=2)

    save_grid(images_before, os.path.join(PLOTS_DIR, "samples_before_rl.png"),
              f"Before DDPO RL (internal-reward arm) — ext.judge mean reward={summary['external_judge_reward_mean_before']:.3f}")
    save_grid(images_after, os.path.join(PLOTS_DIR, "samples_after_rl.png"),
              f"After DDPO RL (internal-reward arm) — ext.judge mean reward={summary['external_judge_reward_mean_after']:.3f}")

    plot_probe_loss(os.path.join(CURVES_DIR, "probe_train.jsonl"),
                     os.path.join(PLOTS_DIR, "probe_train_loss.png"))
    plot_ddpo_reward_and_kl(os.path.join(CURVES_DIR, "ddpo_internal_train.jsonl"),
                             os.path.join(PLOTS_DIR, "ddpo_internal_reward_curve.png"),
                             os.path.join(PLOTS_DIR, "ddpo_internal_kl_to_base.png"),
                             "internal-reward treatment arm")


if __name__ == "__main__":
    main()
