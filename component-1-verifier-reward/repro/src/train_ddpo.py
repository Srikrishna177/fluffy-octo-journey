"""DDPO (Black et al. 2023) policy-gradient fine-tuning at toy scale.

Algorithm 1, vectorized over timesteps:
  1. Sample a batch of full denoising trajectories from the current policy
     (no grad): x_T -> x_{T-1} -> ... -> x_0, storing each reverse transition's
     mean/std implicitly via the stored (x_t, x_{t-1}) pair and the log-prob of
     x_{t-1} under the sampling-time policy.
  2. Score the FINAL image x_0 with the frozen verifier/classifier -> terminal
     reward r. This is the only reward signal (matches DDPO: reward is terminal,
     broadcast to every step for credit assignment).
  3. Standardize r across the batch -> advantage A (same value broadcast to all
     T steps of a trajectory, since the paper's reward is terminal-only).
  4. For num_inner_epochs (default 1, i.e. vanilla policy gradient / REINFORCE;
     >1 uses a PPO-style clipped importance-sampling ratio against the stored
     old log-probs, per the paper's "a few inner epochs" option):
       recompute each transition's log-prob under the CURRENT policy params,
       form ratio = exp(new_logprob - old_logprob), loss = -min(ratio*A,
       clip(ratio, 1-eps, 1+eps)*A), backprop, step.
"""
import argparse
import copy
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from classifier import SmallCNN, reward_fn
from diffusion import DiffusionConfig, GaussianDiffusion
from unet import TinyUNet
from utils import StepLogger, WallClock, set_seed

REPRO_DIR = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")


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
    """Closed-form per-step diagonal-Gaussian KL(policy || reference) on a
    stored trajectory (both models use the same fixed posterior variance
    schedule, so KL reduces to squared mean difference / (2*var))."""
    T, B = x_t_all.shape[0], x_t_all.shape[1]
    H, W = x_t_all.shape[-2], x_t_all.shape[-1]
    t_flat = timesteps.repeat_interleave(B)
    x_t_flat = x_t_all.reshape(T * B, 1, H, W)
    with torch.no_grad():
        mean_p, std_p = diffusion.p_mean_variance(policy, x_t_flat, t_flat)
        mean_r, _ = diffusion.p_mean_variance(reference, x_t_flat, t_flat)
        var = std_p ** 2
        kl = 0.5 * ((mean_p - mean_r) ** 2 / var)
        kl_per_transition = kl.flatten(1).sum(dim=1)  # [T*B]
    return kl_per_transition.view(T, B).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_sec", type=float, default=2700)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--target_class", type=int, default=7)
    ap.add_argument("--num_inner_epochs", type=int, default=1)
    ap.add_argument("--clip_eps", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt_every_sec", type=float, default=120)
    ap.add_argument("--kl_every", type=int, default=5)
    ap.add_argument("--base_ckpt", type=str, default=os.path.join(CKPT_DIR, "ddpm_base.pt"))
    ap.add_argument("--classifier_ckpt", type=str, default=os.path.join(CKPT_DIR, "classifier.pt"))
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

    classifier = SmallCNN()
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location="cpu"))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    logger = StepLogger(
        os.path.join(LOGS_DIR, "curves", "ddpo_train.jsonl"),
        os.path.join(LOGS_DIR, "ddpo_train.log"),
    )
    logger.log(event="start", args=vars(args))

    clock = WallClock(args.budget_sec)
    ckpt_path = os.path.join(CKPT_DIR, "ddpo_finetuned.pt")
    last_ckpt_t = 0.0
    it = 0

    while not clock.expired():
        policy.eval()
        traj = diffusion.sample_trajectory(policy, args.batch_size, img_size=16)
        policy.train()

        reward = reward_fn(classifier, traj["x0_final"], args.target_class)  # [B]
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
    print(f"[ddpo] done. iters={it} elapsed={elapsed:.1f}s")
    print(f"[ddpo] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
