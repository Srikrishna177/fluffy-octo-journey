"""Progressive Distillation (Salimans & Ho, 2202.00512) toy-scale reproduction.

Teacher: Component 1's frozen TinyUNet DDPM (ddpm_base_teacher.pt, ~309k
params, MNIST-16x16, trained ancestral T=50) -- loaded read-only, sampled with
a deterministic DDIM sampler instead of ancestral sampling.

Cascade: 32 -> 16 -> 8 -> 4 -> 2 -> 1 DDIM steps. Each round:
  - student := deep copy of current teacher's weights
  - train student so ONE student DDIM step (t_i -> t_{i+2} in the current
    32/16/8/4/2-step schedule) matches what the current teacher's TWO
    consecutive DDIM steps (t_i -> t_{i+1} -> t_{i+2}) produce, from the same
    starting point.
  - parameterization: paper's Algorithm-2 style analytic target. Training
    points z_{t_i} = alpha_{t_i}*x0 + sigma_{t_i}*eps are drawn the same way
    as DDPM training (real MNIST x0, random eps, random t_i on the CURRENT
    schedule) -- this matches the paper's own training procedure (it samples
    from p_data, not from full teacher rollouts from pure noise).
  - loss: MSE between student's implied x0 prediction (from its single eps
    call at t_i) and the analytically-inverted x0 target implied by the
    teacher's two-step composite landing point z_{t_i+2}. This is Algorithm 2
    of the paper (x-parameterized distillation loss, unweighted -- the
    paper's SNR truncated weighting is skipped for toy-scale simplicity; see
    RUN_LOG.md).
  - student becomes next round's teacher.
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from classifier import SmallCNN
from data import load_mnist16
from diffusion import DiffusionConfig, GaussianDiffusion, halve_schedule, make_ddim_schedule
from unet import TinyUNet
from utils import StepLogger, WallClock, count_params, set_seed

REPRO_DIR = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")
CURVES_DIR = os.path.join(LOGS_DIR, "curves")


@torch.no_grad()
def teacher_two_step_target(diffusion, teacher, z_ti, t_i, t_mid, t_end):
    """z_ti: [B,1,H,W]; t_i, t_mid: python ints (shared within a group);
    t_end: python int or None (terminal -> x0). Returns z_target [B,1,H,W]
    at time t_end (or x0)."""
    b = z_ti.shape[0]
    z_mid, _, _ = diffusion.ddim_step(teacher, z_ti, t_i, t_mid)
    z_end, _, _ = diffusion.ddim_step(teacher, z_mid, t_mid, t_end)
    return z_end


def invert_x_target(diffusion, z_ti, z_end, t_i, t_end):
    """Analytic inversion (paper Algorithm 2): given the two-step teacher
    landing point z_end at time t_end (or x0 if t_end is None), solve for the
    x0 a SINGLE DDIM step from z_ti at t_i would need to reproduce it exactly:
        z_end = alpha_end * x_target + sigma_end * eps_used
        z_ti  = alpha_ti  * x_target + sigma_ti  * eps_used   (same eps_used,
                                                                DDIM is an ODE)
    => x_target = (z_end - (alpha_end/alpha_ti) * z_ti) / (sigma_end - (alpha_end/alpha_ti) * sigma_ti)
    """
    alpha_ti = diffusion.sqrt_alphas_cumprod[t_i]
    sigma_ti = diffusion.sqrt_one_minus_alphas_cumprod[t_i]
    if t_end is None:
        alpha_end = torch.tensor(1.0)
        sigma_end = torch.tensor(0.0)
    else:
        alpha_end = diffusion.sqrt_alphas_cumprod[t_end]
        sigma_end = diffusion.sqrt_one_minus_alphas_cumprod[t_end]
    ratio = alpha_end / alpha_ti
    denom = sigma_end - ratio * sigma_ti
    x_target = (z_end - ratio * z_ti) / denom
    return torch.clamp(x_target, -3.0, 3.0)  # loose clamp, purely numerical-safety


def train_round(diffusion, teacher, student, old_schedule, new_schedule, train_imgs,
                 budget_sec, max_steps, batch_size, lr, logger, round_idx, seed):
    """One halving round: old_schedule (len S) -> new_schedule (len S/2)."""
    set_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    clock = WallClock(budget_sec)
    n_pairs = len(new_schedule)  # == len(old_schedule)//2
    n_data = train_imgs.shape[0]
    step = 0
    losses = []
    t0 = time.time()
    while not clock.expired() and step < max_steps:
        # sample a batch of real MNIST x0 (paper's training distribution: p_data)
        data_idx = torch.randint(0, n_data, (batch_size,))
        x0 = train_imgs[data_idx]
        # sample which coarse pair (schedule index) each example distills
        pair_idx = torch.randint(0, n_pairs, (batch_size,))
        eps = torch.randn_like(x0)

        loss_total = 0.0
        n_groups = 0
        opt.zero_grad()
        # group by pair index (few distinct values -> cheap python loop, exact math)
        for p in range(n_pairs):
            mask = pair_idx == p
            n_sel = int(mask.sum().item())
            if n_sel == 0:
                continue
            t_i = old_schedule[2 * p]
            t_mid = old_schedule[2 * p + 1]
            t_end = old_schedule[2 * p + 2] if (2 * p + 2) < len(old_schedule) else None

            x0_sel = x0[mask]
            eps_sel = eps[mask]
            t_i_batch = torch.full((n_sel,), t_i, dtype=torch.long)
            z_ti = diffusion.q_sample(x0_sel, t_i_batch, eps_sel)

            with torch.no_grad():
                z_end = teacher_two_step_target(diffusion, teacher, z_ti, t_i, t_mid, t_end)
            x_target = invert_x_target(diffusion, z_ti, z_end, t_i, t_end)

            eps_student = student(z_ti, t_i_batch)
            alpha_ti = diffusion.sqrt_alphas_cumprod[t_i]
            sigma_ti = diffusion.sqrt_one_minus_alphas_cumprod[t_i]
            x0_student = (z_ti - sigma_ti * eps_student) / alpha_ti

            loss_p = torch.mean((x0_student - x_target) ** 2) * n_sel
            loss_total = loss_total + loss_p
            n_groups += n_sel

        loss = loss_total / max(n_groups, 1)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 20 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            logger.log(event="step", round=round_idx, step=step,
                        loss=round(loss.item(), 6), avg_loss_50=round(avg, 6),
                        elapsed_sec=round(clock.elapsed(), 1))
        step += 1

    elapsed = time.time() - t0
    final_avg = sum(losses[-50:]) / len(losses[-50:]) if losses else float("nan")
    logger.log(event="round_done", round=round_idx, steps=step, elapsed_sec=round(elapsed, 1),
               final_avg_loss=round(final_avg, 6),
               old_nfe=len(old_schedule), new_nfe=len(new_schedule))
    return {"steps": step, "elapsed_sec": elapsed, "final_avg_loss": final_avg}


@torch.no_grad()
def eval_arm(diffusion, model, schedule, fixed_noise, classifier, teacher_samples_32=None):
    """Sample from `model` on `schedule` starting from the SAME fixed_noise
    used by every arm; measure wall-clock, compute trajectory-fidelity vs.
    teacher_samples_32 (if given) and absolute classifier-judge quality."""
    b = fixed_noise.shape[0]
    t_start = time.time()
    x = fixed_noise.clone()
    for i, t_cur in enumerate(schedule):
        t_next = schedule[i + 1] if i + 1 < len(schedule) else None
        x, _, _ = diffusion.ddim_step(model, x, t_cur, t_next)
    wall_sec = time.time() - t_start

    logits = classifier(x)
    probs = torch.softmax(logits, dim=-1)
    top1_conf = probs.max(dim=-1).values
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)

    out = {
        "nfe": len(schedule),
        "wall_clock_sec": wall_sec,
        "wall_clock_sec_per_sample": wall_sec / b,
        "own_quality_mean_top1_conf": top1_conf.mean().item(),
        "own_quality_mean_entropy": entropy.mean().item(),
        "samples": x,
    }
    if teacher_samples_32 is not None:
        pixel_mse = torch.mean((x - teacher_samples_32) ** 2).item()
        t_logits = classifier(teacher_samples_32)
        t_probs = torch.softmax(t_logits, dim=-1)
        l2_dist = torch.mean(torch.sum((probs - t_probs) ** 2, dim=-1)).item()
        kl_dist = torch.mean(
            torch.sum(t_probs * (torch.log(t_probs.clamp_min(1e-9)) - torch.log(probs.clamp_min(1e-9))), dim=-1)
        ).item()
        out.update({
            "trajectory_pixel_mse_vs_teacher32": pixel_mse,
            "trajectory_classifier_l2_vs_teacher32": l2_dist,
            "trajectory_classifier_kl_vs_teacher32": kl_dist,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--base_nfe", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--round_budget_sec", type=float, default=360)
    ap.add_argument("--round_max_steps", type=int, default=1500)
    ap.add_argument("--total_budget_sec", type=float, default=3300)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--eval_seed", type=int, default=12345)
    ap.add_argument("--collapse_ratio", type=float, default=3.0,
                     help="stop cascade if a round's trajectory-MSE is > this x the previous round's")
    ap.add_argument("--collapse_conf_floor", type=float, default=0.15,
                     help="stop cascade if own-quality mean top1 confidence collapses below this (~random=0.1 for 10 classes)")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)

    set_seed(0)
    diffusion = GaussianDiffusion(DiffusionConfig(T=args.T))

    classifier = SmallCNN()
    classifier.load_state_dict(torch.load(os.path.join(CKPT_DIR, "classifier.pt"), map_location="cpu"))
    classifier.eval()

    teacher = TinyUNet()
    teacher.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ddpm_base_teacher.pt"), map_location="cpu"))
    teacher.eval()
    n_params = count_params(teacher)
    print(f"[distill] teacher params={n_params}", flush=True)

    train_ds, _, used_real = load_mnist16(args.n_train, 200, seed=0)
    train_imgs = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    print(f"[distill] train images={train_imgs.shape[0]} used_real_mnist={used_real}", flush=True)

    logger = StepLogger(
        os.path.join(CURVES_DIR, "distill_train.jsonl"),
        os.path.join(LOGS_DIR, "distill_train.log"),
    )
    logger.log(event="start", n_params=n_params, used_real_mnist=used_real, args=vars(args))

    schedule_32 = make_ddim_schedule(args.T, args.base_nfe)
    assert len(schedule_32) == args.base_nfe, f"schedule collision: got {len(schedule_32)} unique steps, wanted {args.base_nfe}"

    # fixed eval noise, shared across every arm (fair trajectory comparison)
    gen = torch.Generator().manual_seed(args.eval_seed)
    fixed_noise = torch.randn(args.n_eval, 1, 16, 16, generator=gen)

    results = {}
    plots_samples = {}

    # ---- arm 0: frozen teacher, 32-step DDIM baseline (no training) ----
    teacher_eval = eval_arm(diffusion, teacher, schedule_32, fixed_noise, classifier, teacher_samples_32=None)
    teacher_samples_32 = teacher_eval.pop("samples")
    teacher_eval["trajectory_pixel_mse_vs_teacher32"] = 0.0
    teacher_eval["trajectory_classifier_l2_vs_teacher32"] = 0.0
    teacher_eval["trajectory_classifier_kl_vs_teacher32"] = 0.0
    results[32] = teacher_eval
    plots_samples[32] = teacher_samples_32
    print(f"[distill] arm nfe=32 (frozen teacher, no training): {teacher_eval}", flush=True)

    total_clock = WallClock(args.total_budget_sec)
    current_teacher = teacher
    current_schedule = schedule_32
    prev_mse = None
    stop_reason = None
    round_idx = 0

    while len(current_schedule) > 1:
        if total_clock.expired():
            stop_reason = f"total cascade budget ({args.total_budget_sec}s) exhausted before round to {len(current_schedule)//2} steps"
            print(f"[distill] STOP: {stop_reason}", flush=True)
            break

        round_idx += 1
        new_schedule = halve_schedule(current_schedule)
        student = TinyUNet()
        student.load_state_dict(copy.deepcopy(current_teacher.state_dict()))  # init = copy of teacher

        remaining = total_clock.remaining()
        this_budget = min(args.round_budget_sec, max(remaining, 0.0))
        print(f"[distill] round {round_idx}: {len(current_schedule)} -> {len(new_schedule)} steps, budget={this_budget:.0f}s", flush=True)

        stats = train_round(
            diffusion, current_teacher, student, current_schedule, new_schedule,
            train_imgs, budget_sec=this_budget, max_steps=args.round_max_steps,
            batch_size=args.batch_size, lr=args.lr, logger=logger, round_idx=round_idx, seed=100 + round_idx,
        )
        student.eval()

        ckpt_path = os.path.join(CKPT_DIR, f"student_{len(new_schedule)}step.pt")
        torch.save(student.state_dict(), ckpt_path)

        arm_eval = eval_arm(diffusion, student, new_schedule, fixed_noise, classifier, teacher_samples_32=teacher_samples_32)
        samples = arm_eval.pop("samples")
        arm_eval["train_steps"] = stats["steps"]
        arm_eval["train_elapsed_sec"] = stats["elapsed_sec"]
        arm_eval["train_final_avg_loss"] = stats["final_avg_loss"]
        results[len(new_schedule)] = arm_eval
        plots_samples[len(new_schedule)] = samples
        print(f"[distill] arm nfe={len(new_schedule)}: {arm_eval}", flush=True)

        cur_mse = arm_eval["trajectory_pixel_mse_vs_teacher32"]
        cur_conf = arm_eval["own_quality_mean_top1_conf"]
        collapsed = False
        if prev_mse is not None and cur_mse > args.collapse_ratio * prev_mse and cur_mse > 0.02:
            collapsed = True
            stop_reason = (f"trajectory-MSE collapsed at {len(new_schedule)} steps: "
                            f"{cur_mse:.5f} > {args.collapse_ratio}x previous round's {prev_mse:.5f}")
        if cur_conf < args.collapse_conf_floor:
            collapsed = True
            stop_reason = (f"classifier-judge confidence collapsed to near-random at {len(new_schedule)} steps: "
                            f"mean top1 conf={cur_conf:.3f} < floor {args.collapse_conf_floor}")

        prev_mse = cur_mse
        current_teacher = student
        current_schedule = new_schedule

        if collapsed:
            print(f"[distill] STOP (quality collapse): {stop_reason}", flush=True)
            break

    if stop_reason is None:
        stop_reason = "cascade completed all rounds down to NFE=1 within budget, no collapse triggered"

    logger.log(event="cascade_done", stop_reason=stop_reason, rounds_completed=round_idx,
               total_elapsed_sec=round(total_clock.elapsed(), 1))

    with open(os.path.join(LOGS_DIR, "results.json"), "w") as f:
        json.dump({
            "stop_reason": stop_reason,
            "rounds_completed": round_idx,
            "total_elapsed_sec": total_clock.elapsed(),
            "teacher_params": n_params,
            "used_real_mnist": used_real,
            "schedule_32": schedule_32,
            "results_by_nfe": results,
        }, f, indent=2)

    torch.save(plots_samples, os.path.join(CKPT_DIR, "eval_samples_by_nfe.pt"))
    print(f"[distill] DONE. stop_reason={stop_reason}", flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
