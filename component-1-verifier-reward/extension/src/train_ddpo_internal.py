"""DDPO fine-tuning, TREATMENT arm: identical Algorithm 1 implementation and
hyperparameters to repro/src/train_ddpo.py — same TinyUNet policy init, same
T=50, batch_size=48, lr=1e-4, num_inner_epochs=1, clip_eps=1e-4, target
class 7, same PPO-surrogate/REINFORCE update, same KL-to-base estimate.

The ONLY change from the reproduction: reward = internal-probe confidence
(frozen DDPM bottleneck features -> frozen linear probe -> softmax P(class=7))
instead of the frozen external CNN classifier's confidence. The feature
extractor and probe are both frozen throughout (no gradient into either);
exactly like the external classifier, the only thing DDPO ever backprops
through is the policy's own log-probs.
"""
import argparse
import copy
import os
import sys

import torch

EXT_DIR = os.path.dirname(__file__)
REPRO_SRC = os.path.join(EXT_DIR, "..", "..", "repro", "src")
sys.path.insert(0, REPRO_SRC)
sys.path.insert(0, EXT_DIR)

from diffusion import DiffusionConfig, GaussianDiffusion  # noqa: E402
from unet import TinyUNet  # noqa: E402
from utils import StepLogger, WallClock, set_seed  # noqa: E402
from probe import FeatureExtractor, LinearProbe, internal_reward_fn  # noqa: E402

EXT_ROOT = os.path.join(EXT_DIR, "..")
LOGS_DIR = os.path.join(EXT_ROOT, "logs")
CKPT_DIR = os.path.join(EXT_ROOT, "checkpoints")
REPRO_CKPT_DIR = os.path.join(EXT_DIR, "..", "..", "repro", "checkpoints")


def compute_log_probs(diffusion, model, x_t_all, x_prev_all, timesteps):
    T, B = x_t_all.shape[0], x_t_all.shape[1]
    H, W = x_t_all.shape[-2], x_t_all.shape[-1]
    t_flat = timesteps.repeat_interleave(B)
    x_t_flat = x_t_all.reshape(T * B, 1, H, W)
    x_prev_flat = x_prev_all.reshape(T * B, 1, H, W)
    mean, std = diffusion.p_mean_variance(model, x_t_flat, t_flat)
    lp = diffusion.gaussian_log_prob(x_prev_flat, mean, std)
    return lp.view(T, B)


def kl_to_base(diffusion, policy, reference, x_t_all, timesteps):
    T, B = x_t_all.shape[0], x_t_all.shape[1]
    H, W = x_t_all.shape[-2], x_t_all.shape[-1]
    t_flat = timesteps.repeat_interleave(B)
    x_t_flat = x_t_all.reshape(T * B, 1, H, W)
    with torch.no_grad():
        mean_p, std_p = diffusion.p_mean_variance(policy, x_t_flat, t_flat)
        mean_r, _ = diffusion.p_mean_variance(reference, x_t_flat, t_flat)
        var = std_p ** 2
        kl = 0.5 * ((mean_p - mean_r) ** 2 / var)
        kl_per_transition = kl.flatten(1).sum(dim=1)
    return kl_per_transition.view(T, B).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_sec", type=float, default=1200)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--target_class", type=int, default=7)
    ap.add_argument("--num_inner_epochs", type=int, default=1)
    ap.add_argument("--clip_eps", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt_every_sec", type=float, default=120)
    ap.add_argument("--kl_every", type=int, default=5)
    ap.add_argument("--base_ckpt", type=str, default=os.path.join(REPRO_CKPT_DIR, "ddpm_base.pt"))
    ap.add_argument("--probe_ckpt", type=str, default=os.path.join(CKPT_DIR, "probe.pt"))
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    diffusion = GaussianDiffusion(DiffusionConfig(T=args.T))

    policy = TinyUNet()
    policy.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))

    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    # Internal verifier: frozen copy of the SAME pretrained base UNet, used
    # only as a feature extractor (mid-bottleneck forward hook), plus the
    # frozen linear probe trained in train_probe.py. Never updated by DDPO.
    feat_backbone = copy.deepcopy(policy)
    extractor = FeatureExtractor(feat_backbone)
    probe = LinearProbe()
    probe.load_state_dict(torch.load(args.probe_ckpt, map_location="cpu"))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    logger = StepLogger(
        os.path.join(LOGS_DIR, "curves", "ddpo_internal_train.jsonl"),
        os.path.join(LOGS_DIR, "ddpo_internal_train.log"),
    )
    logger.log(event="start", args=vars(args), reward_source="internal_probe")

    clock = WallClock(args.budget_sec)
    ckpt_path = os.path.join(CKPT_DIR, "ddpo_internal_finetuned.pt")
    last_ckpt_t = 0.0
    it = 0

    while not clock.expired():
        policy.eval()
        traj = diffusion.sample_trajectory(policy, args.batch_size, img_size=16)
        policy.train()

        reward = internal_reward_fn(extractor, probe, traj["x0_final"], args.target_class)  # [B]
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)  # [B]
        adv_bcast = adv.unsqueeze(0).expand(args.T, args.batch_size)  # [T,B]

        old_log_probs = traj["old_log_probs"].detach()
        last_loss = None
        for inner in range(args.num_inner_epochs):
            new_log_probs = compute_log_probs(diffusion, policy, traj["x_t"], traj["x_prev"], traj["timesteps"])
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * adv_bcast
            clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_bcast
            loss = -torch.mean(torch.min(unclipped, clipped))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            last_loss = loss.item()

        log_kwargs = dict(
            event="iter", it=it,
            reward_mean=round(reward.mean().item(), 4),
            reward_std=round(reward.std().item(), 4),
            loss=round(last_loss, 6),
            elapsed_sec=round(clock.elapsed(), 1),
        )
        if it % args.kl_every == 0:
            kl = kl_to_base(diffusion, policy, reference, traj["x_t"], traj["timesteps"])
            log_kwargs["kl_to_base"] = round(kl, 4)
        logger.log(**log_kwargs)

        if clock.elapsed() - last_ckpt_t > args.ckpt_every_sec:
            torch.save(policy.state_dict(), ckpt_path)
            last_ckpt_t = clock.elapsed()

        it += 1

    torch.save(policy.state_dict(), ckpt_path)
    elapsed = clock.elapsed()
    logger.log(event="done", elapsed_sec=round(elapsed, 1), total_iters=it)
    print(f"[ddpo-internal] done. iters={it} elapsed={elapsed:.1f}s")
    print(f"[ddpo-internal] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
