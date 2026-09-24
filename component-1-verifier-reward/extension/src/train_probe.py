"""Train the internal-verifier linear probe on frozen bottleneck features of
the already-pretrained, frozen DDPM UNet (repro/checkpoints/ddpm_base.pt).

The UNet is loaded once, frozen, and never updated. Real images are passed
through q_sample at the fixed near-zero timestep FEATURE_TIMESTEP=0 (adding
the (very small) t=0 noise the UNet was trained to expect at that step),
then through the frozen UNet to read out the `mid` bottleneck activation via
a forward hook (probe.FeatureExtractor). Only a single linear layer on top
is trained, by cross-entropy against real MNIST-16x16 labels, using the same
data pipeline as the reproduction (repro/src/data.py).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

EXT_DIR = os.path.dirname(__file__)
REPRO_SRC = os.path.join(EXT_DIR, "..", "..", "repro", "src")
sys.path.insert(0, REPRO_SRC)
sys.path.insert(0, EXT_DIR)

from data import load_mnist16  # noqa: E402
from diffusion import DiffusionConfig, GaussianDiffusion  # noqa: E402
from unet import TinyUNet  # noqa: E402
from utils import StepLogger, count_params, set_seed  # noqa: E402
from probe import FeatureExtractor, LinearProbe, FEATURE_TIMESTEP  # noqa: E402

EXT_ROOT = os.path.join(EXT_DIR, "..")
LOGS_DIR = os.path.join(EXT_ROOT, "logs")
CKPT_DIR = os.path.join(EXT_ROOT, "checkpoints")
REPRO_CKPT_DIR = os.path.join(EXT_DIR, "..", "..", "repro", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_ckpt", type=str, default=os.path.join(REPRO_CKPT_DIR, "ddpm_base.pt"))
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_ds, test_ds, used_real = load_mnist16(args.n_train, args.n_test, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    diffusion = GaussianDiffusion(DiffusionConfig(T=50))

    unet = TinyUNet()
    unet.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    extractor = FeatureExtractor(unet)  # frozen, eval mode, hooked on `mid`

    probe = LinearProbe()
    n_params = count_params(probe)
    print(f"[probe] linear-probe params={n_params} used_real_mnist={used_real} "
          f"feature_timestep={FEATURE_TIMESTEP}", flush=True)

    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    logger = StepLogger(
        os.path.join(LOGS_DIR, "curves", "probe_train.jsonl"),
        os.path.join(LOGS_DIR, "probe_train.log"),
    )
    logger.log(event="start", n_params=n_params, used_real_mnist=used_real,
               feature_timestep=FEATURE_TIMESTEP, args=vars(args))

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        probe.train()
        for imgs, labels in train_loader:
            b = imgs.shape[0]
            t0_batch = torch.zeros(b, dtype=torch.long)
            noise = torch.randn_like(imgs)
            x_t0 = diffusion.q_sample(imgs, t0_batch, noise)  # near-zero-timestep noised input
            with torch.no_grad():
                feat = extractor.extract(x_t0, FEATURE_TIMESTEP)
            opt.zero_grad()
            logits = probe(feat)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            if step % 10 == 0:
                acc = (logits.argmax(-1) == labels).float().mean().item()
                logger.log(event="step", epoch=epoch, step=step,
                           loss=round(loss.item(), 4), train_acc=round(acc, 4))
            step += 1

        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                b = imgs.shape[0]
                t0_batch = torch.zeros(b, dtype=torch.long)
                noise = torch.randn_like(imgs)
                x_t0 = diffusion.q_sample(imgs, t0_batch, noise)
                feat = extractor.extract(x_t0, FEATURE_TIMESTEP)
                logits = probe(feat)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.numel()
        test_acc = correct / total
        logger.log(event="epoch_end", epoch=epoch, test_acc=round(test_acc, 4),
                   elapsed_sec=round(time.time() - t0, 1))

    elapsed = time.time() - t0
    logger.log(event="done", elapsed_sec=round(elapsed, 1), final_test_acc=round(test_acc, 4))

    ckpt_path = os.path.join(CKPT_DIR, "probe.pt")
    torch.save(probe.state_dict(), ckpt_path)
    print(f"[probe] saved checkpoint to {ckpt_path}, wall_clock={elapsed:.1f}s, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
