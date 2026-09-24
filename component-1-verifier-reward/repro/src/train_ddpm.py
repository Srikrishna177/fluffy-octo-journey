"""Pretrain the base DDPM (TinyUNet, epsilon-prediction, T=50) on MNIST-16x16,
unconditional, from scratch. Time-boxed to ~45-60 min wall clock (spec);
checkpoints periodically so a cut-short run still leaves a usable model.
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import load_mnist16
from diffusion import DiffusionConfig, GaussianDiffusion
from unet import TinyUNet
from utils import StepLogger, WallClock, count_params, set_seed

REPRO_DIR = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_sec", type=float, default=2700)  # 45 min default
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_every_sec", type=float, default=120)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_ds, _, used_real = load_mnist16(args.n_train, args.n_test, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    diffusion = GaussianDiffusion(DiffusionConfig(T=args.T))
    model = TinyUNet()
    n_params = count_params(model)
    print(f"[ddpm] params={n_params} used_real_mnist={used_real}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger = StepLogger(
        os.path.join(LOGS_DIR, "curves", "ddpm_train.jsonl"),
        os.path.join(LOGS_DIR, "ddpm_train.log"),
    )
    logger.log(event="start", n_params=n_params, used_real_mnist=used_real, args=vars(args))

    clock = WallClock(args.budget_sec)
    ckpt_path = os.path.join(CKPT_DIR, "ddpm_base.pt")
    last_ckpt_t = 0.0
    step = 0
    epoch = 0
    running_loss = []

    while not clock.expired():
        epoch += 1
        for imgs, _ in train_loader:
            if clock.expired():
                break
            opt.zero_grad()
            b = imgs.shape[0]
            t = torch.randint(0, args.T, (b,), dtype=torch.long)
            noise = torch.randn_like(imgs)
            x_t = diffusion.q_sample(imgs, t, noise)
            eps_pred = model(x_t, t)
            loss = torch.mean((eps_pred - noise) ** 2)
            loss.backward()
            opt.step()

            running_loss.append(loss.item())
            if step % 20 == 0:
                avg = sum(running_loss[-100:]) / len(running_loss[-100:])
                logger.log(
                    event="step", epoch=epoch, step=step,
                    loss=round(loss.item(), 5), avg_loss_100=round(avg, 5),
                    elapsed_sec=round(clock.elapsed(), 1),
                )
            step += 1

            if clock.elapsed() - last_ckpt_t > args.ckpt_every_sec:
                torch.save(model.state_dict(), ckpt_path)
                last_ckpt_t = clock.elapsed()

    torch.save(model.state_dict(), ckpt_path)
    elapsed = clock.elapsed()
    final_avg = sum(running_loss[-100:]) / len(running_loss[-100:]) if running_loss else float("nan")
    logger.log(event="done", elapsed_sec=round(elapsed, 1), total_steps=step, total_epochs=epoch, final_avg_loss_100=round(final_avg, 5))
    print(f"[ddpm] done. steps={step} epochs={epoch} elapsed={elapsed:.1f}s final_avg_loss={final_avg:.5f}")
    print(f"[ddpm] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
