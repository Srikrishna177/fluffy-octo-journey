"""Extension: on-policy training-state generation for progressive distillation
(direct test of the reproduction's exposure-bias diagnosis).

Recap (see ../DECISION.md and ../../logs/RUN_LOG.md): the reproduction found
that the first halving round (32->16 DDIM steps) of progressive distillation
collapsed immediately -- the 16-step student mode-collapsed onto a single
classifier-favored digit class for the whole fixed-noise eval batch, even
though the paper's own Algorithm-2 regression loss decreased sensibly. The
RUN_LOG hypothesizes this is exposure bias: Algorithm 2 trains on noisy
states z_t drawn from the DATA marginal (real MNIST x0 + fresh noise), never
from the model's own actual sampling rollout starting at pure noise -- a
train/test mismatch the student can exploit with a cheap "always predict one
confident digit" shortcut.

This script keeps EVERYTHING from the reproduction's round-1 (32->16) run
fixed -- same frozen teacher, same 32-step DDIM schedule, same student
architecture/init (deep copy of the teacher), same batch_size/lr/round
budget/max_steps, same regression-target computation
(`teacher_two_step_target` / `invert_x_target`, imported unchanged from
`repro/src/distill.py`) -- and changes ONLY where the training input states
z_ti come from: instead of q_sample(x0_real_mnist, t_i, eps_fresh), they are
states actually visited by the teacher's own 32-step DDIM rollout starting
from pure noise (a "replay pool" of on-policy states, built once up front,
then sampled from during training exactly like the data-marginal version
samples from its 10,000-image MNIST pool).
"""
import argparse
import copy
import json
import os
import sys
import time

import torch

REPRO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "repro", "src")
sys.path.insert(0, os.path.abspath(REPRO_SRC))

from classifier import SmallCNN  # noqa: E402
from diffusion import DiffusionConfig, GaussianDiffusion, halve_schedule, make_ddim_schedule  # noqa: E402
from distill import eval_arm, invert_x_target, teacher_two_step_target  # noqa: E402  (REUSED, not reimplemented)
from unet import TinyUNet  # noqa: E402
from utils import StepLogger, WallClock, count_params, set_seed  # noqa: E402

EXT_DIR = os.path.join(os.path.dirname(__file__), "..")
REPRO_DIR = os.path.join(EXT_DIR, "..", "repro")
REPRO_CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")
REPRO_LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
LOGS_DIR = os.path.join(EXT_DIR, "logs")
CKPT_DIR = os.path.join(EXT_DIR, "checkpoints")
CURVES_DIR = os.path.join(LOGS_DIR, "curves")


@torch.no_grad()
def generate_onpolicy_pool(diffusion, teacher, schedule, pool_size, batch_size, seed):
    """Roll the frozen teacher out from pure noise on its own 32-step DDIM
    schedule; record the state actually visited at every EVEN schedule index
    (i.e. every t_i a training pair starts from: t_i = schedule[2*p]) before
    taking that pair's two steps. Returns a list of n_pairs tensors, each
    [pool_size, 1, 16, 16] -- the on-policy replacement for the data-marginal
    q_sample(x0, t_i, eps) pool.
    """
    set_seed(seed)
    n_pairs = len(schedule) // 2
    pool = [[] for _ in range(n_pairs)]
    n_done = 0
    n_batches = 0
    t0 = time.time()
    while n_done < pool_size:
        b = min(batch_size, pool_size - n_done)
        x = torch.randn(b, 1, 16, 16)
        for i, t_cur in enumerate(schedule):
            if i % 2 == 0:
                pool[i // 2].append(x.clone())
            t_next = schedule[i + 1] if i + 1 < len(schedule) else None
            x, _, _ = diffusion.ddim_step(teacher, x, t_cur, t_next)
        n_done += b
        n_batches += 1
    pool = [torch.cat(p, dim=0)[:pool_size] for p in pool]
    elapsed = time.time() - t0
    return pool, elapsed, n_batches


def train_round_onpolicy(diffusion, teacher, student, old_schedule, new_schedule, pool,
                          budget_sec, max_steps, batch_size, lr, logger, round_idx, seed):
    """Identical training loop / hyperparameter handling to repro/src/distill.py's
    train_round, EXCEPT z_ti is drawn from the precomputed on-policy `pool`
    (indexed by pair) instead of q_sample(real_mnist_x0, t_i, fresh_eps)."""
    set_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    clock = WallClock(budget_sec)
    n_pairs = len(new_schedule)
    pool_size = pool[0].shape[0]
    step = 0
    losses = []
    t0 = time.time()
    while not clock.expired() and step < max_steps:
        pair_idx = torch.randint(0, n_pairs, (batch_size,))

        loss_total = 0.0
        n_groups = 0
        opt.zero_grad()
        for p in range(n_pairs):
            mask = pair_idx == p
            n_sel = int(mask.sum().item())
            if n_sel == 0:
                continue
            t_i = old_schedule[2 * p]
            t_mid = old_schedule[2 * p + 1]
            t_end = old_schedule[2 * p + 2] if (2 * p + 2) < len(old_schedule) else None

            idx = torch.randint(0, pool_size, (n_sel,))
            z_ti = pool[p][idx]  # <-- ON-POLICY state, in place of q_sample(x0, t_i, eps)

            with torch.no_grad():
                z_end = teacher_two_step_target(diffusion, teacher, z_ti, t_i, t_mid, t_end)
            x_target = invert_x_target(diffusion, z_ti, z_end, t_i, t_end)

            t_i_batch = torch.full((n_sel,), t_i, dtype=torch.long)
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


def population_diversity_metrics(classifier, samples):
    """Same computation used to diagnose the reproduction's collapse
    (predicted-class histogram entropy + majority-class fraction), reimplemented
    here since the reproduction's version was computed inline and not saved as
    a standalone function. See ../../logs/RUN_LOG.md 'Honest diagnosis'."""
    import math
    with torch.no_grad():
        logits = classifier(samples)
        pred = logits.argmax(dim=-1).tolist()
    n = len(pred)
    hist = {}
    for c in pred:
        hist[c] = hist.get(c, 0) + 1
    ent = 0.0
    for c, cnt in hist.items():
        p = cnt / n
        ent -= p * math.log(p)
    maj_frac = max(hist.values()) / n
    return {
        "pred_class_histogram": {str(k): v for k, v in sorted(hist.items())},
        "population_class_entropy_nats": ent,
        "frac_samples_in_majority_class": maj_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--base_nfe", type=int, default=32)
    ap.add_argument("--pool_size", type=int, default=4096,
                     help="number of independent on-policy teacher rollouts (= pool size per pair index)")
    ap.add_argument("--pool_batch", type=int, default=512)
    ap.add_argument("--pool_seed", type=int, default=999)
    # training hyperparameters: MATCH the reproduction's round-1 (32->16) run exactly
    # (repro/src/distill.py defaults / logs/RUN_LOG.md main-cascade command)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--round_budget_sec", type=float, default=300)
    ap.add_argument("--round_max_steps", type=int, default=4000)
    ap.add_argument("--train_seed", type=int, default=101)  # repro round 1 used seed=100+round_idx(1)=101
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--eval_seed", type=int, default=12345)
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)

    set_seed(0)
    diffusion = GaussianDiffusion(DiffusionConfig(T=args.T))

    classifier = SmallCNN()
    classifier.load_state_dict(torch.load(os.path.join(REPRO_CKPT_DIR, "classifier.pt"), map_location="cpu"))
    classifier.eval()

    teacher = TinyUNet()
    teacher.load_state_dict(torch.load(os.path.join(REPRO_CKPT_DIR, "ddpm_base_teacher.pt"), map_location="cpu"))
    teacher.eval()
    n_params = count_params(teacher)
    print(f"[distill_onpolicy] teacher params={n_params} (frozen, loaded read-only)", flush=True)

    logger = StepLogger(
        os.path.join(CURVES_DIR, "distill_train_onpolicy.jsonl"),
        os.path.join(LOGS_DIR, "distill_train_onpolicy.log"),
    )
    logger.log(event="start", n_params=n_params, args=vars(args))

    schedule_32 = make_ddim_schedule(args.T, args.base_nfe)
    assert len(schedule_32) == args.base_nfe
    schedule_16 = halve_schedule(schedule_32)

    # ---- 1. build the on-policy replay pool ----
    wall_total_start = time.time()
    print(f"[distill_onpolicy] generating on-policy pool: pool_size={args.pool_size} "
          f"batch={args.pool_batch} ...", flush=True)
    pool, pool_elapsed, n_pool_batches = generate_onpolicy_pool(
        diffusion, teacher, schedule_32, args.pool_size, args.pool_batch, args.pool_seed)
    pool_sizes = [p.shape[0] for p in pool]
    print(f"[distill_onpolicy] pool built in {pool_elapsed:.1f}s "
          f"({n_pool_batches} rollout batches, {len(pool)} pair-slots, sizes={pool_sizes[:3]}...)", flush=True)
    logger.log(event="pool_done", elapsed_sec=round(pool_elapsed, 1),
               pool_size=args.pool_size, n_pair_slots=len(pool), n_rollout_batches=n_pool_batches)

    # ---- 2. train the 16-step student on on-policy states ----
    set_seed(args.train_seed)
    student = TinyUNet()
    student.load_state_dict(copy.deepcopy(teacher.state_dict()))  # init = copy of teacher, per spec

    print(f"[distill_onpolicy] training round 1 (32->16, on-policy): "
          f"budget={args.round_budget_sec}s max_steps={args.round_max_steps}", flush=True)
    stats = train_round_onpolicy(
        diffusion, teacher, student, schedule_32, schedule_16, pool,
        budget_sec=args.round_budget_sec, max_steps=args.round_max_steps,
        batch_size=args.batch_size, lr=args.lr, logger=logger, round_idx=1, seed=args.train_seed,
    )
    student.eval()
    print(f"[distill_onpolicy] training done: {stats}", flush=True)

    ckpt_path = os.path.join(CKPT_DIR, "student_16step_onpolicy.pt")
    torch.save(student.state_dict(), ckpt_path)

    wall_total = time.time() - wall_total_start
    print(f"[distill_onpolicy] TOTAL on-policy run wall-clock (pool + train): {wall_total:.1f}s "
          f"({wall_total/60.0:.2f} min)", flush=True)
    logger.log(event="onpolicy_run_done", pool_elapsed_sec=round(pool_elapsed, 1),
               train_elapsed_sec=round(stats["elapsed_sec"], 1),
               total_elapsed_sec=round(wall_total, 1))

    # ---- 3. evaluate: teacher (32), data-marginal baseline (16, reused checkpoint), on-policy (16) ----
    gen = torch.Generator().manual_seed(args.eval_seed)
    fixed_noise = torch.randn(args.n_eval, 1, 16, 16, generator=gen)

    teacher_eval = eval_arm(diffusion, teacher, schedule_32, fixed_noise, classifier, teacher_samples_32=None)
    teacher_samples = teacher_eval.pop("samples")
    teacher_eval["trajectory_pixel_mse_vs_teacher32"] = 0.0
    teacher_eval["trajectory_classifier_l2_vs_teacher32"] = 0.0
    teacher_eval["trajectory_classifier_kl_vs_teacher32"] = 0.0
    teacher_eval.update(population_diversity_metrics(classifier, teacher_samples))

    baseline_student = TinyUNet()
    baseline_ckpt = os.path.join(REPRO_CKPT_DIR, "student_16step.pt")
    baseline_student.load_state_dict(torch.load(baseline_ckpt, map_location="cpu"))
    baseline_student.eval()
    baseline_eval = eval_arm(diffusion, baseline_student, schedule_16, fixed_noise, classifier,
                              teacher_samples_32=teacher_samples)
    baseline_samples = baseline_eval.pop("samples")
    baseline_eval.update(population_diversity_metrics(classifier, baseline_samples))

    onpolicy_eval = eval_arm(diffusion, student, schedule_16, fixed_noise, classifier,
                              teacher_samples_32=teacher_samples)
    onpolicy_samples = onpolicy_eval.pop("samples")
    onpolicy_eval["train_steps"] = stats["steps"]
    onpolicy_eval["train_elapsed_sec"] = stats["elapsed_sec"]
    onpolicy_eval["train_final_avg_loss"] = stats["final_avg_loss"]
    onpolicy_eval["pool_gen_elapsed_sec"] = pool_elapsed
    onpolicy_eval["pool_size"] = args.pool_size
    onpolicy_eval.update(population_diversity_metrics(classifier, onpolicy_samples))

    print("[distill_onpolicy] teacher (32-step):", json.dumps({k: v for k, v in teacher_eval.items()}, default=str), flush=True)
    print("[distill_onpolicy] baseline data-marginal (16-step, reused repro checkpoint):",
          json.dumps({k: v for k, v in baseline_eval.items()}, default=str), flush=True)
    print("[distill_onpolicy] on-policy (16-step, this run):",
          json.dumps({k: v for k, v in onpolicy_eval.items()}, default=str), flush=True)

    results = {
        "extension": "on-policy training-state generation (exposure-bias test)",
        "wall_clock_pool_gen_sec": pool_elapsed,
        "wall_clock_train_sec": stats["elapsed_sec"],
        "wall_clock_onpolicy_run_total_sec": wall_total,
        "hyperparameters_matched_to_repro_round1": {
            "batch_size": args.batch_size, "lr": args.lr,
            "round_budget_sec": args.round_budget_sec, "round_max_steps": args.round_max_steps,
            "train_seed": args.train_seed, "student_init": "deep copy of frozen teacher",
        },
        "pool_size": args.pool_size,
        "eval_seed": args.eval_seed,
        "n_eval": args.n_eval,
        "results_by_arm": {
            "teacher_32step": teacher_eval,
            "student_16step_data_marginal": baseline_eval,
            "student_16step_onpolicy": onpolicy_eval,
        },
    }
    with open(os.path.join(LOGS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(
        {"teacher_32": teacher_samples, "student_16_data_marginal": baseline_samples,
         "student_16_onpolicy": onpolicy_samples},
        os.path.join(CKPT_DIR, "eval_samples.pt"),
    )
    print(f"[distill_onpolicy] DONE. wrote {os.path.join(LOGS_DIR, 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
