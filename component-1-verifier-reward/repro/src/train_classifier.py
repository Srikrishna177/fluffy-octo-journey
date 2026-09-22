"""Pretrain the frozen verifier: small CNN classifier on MNIST-16x16.
Cheap prerequisite step (spec: a few minutes).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from classifier import SmallCNN
from data import load_mnist16
from utils import StepLogger, count_params, set_seed

REPRO_DIR = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(REPRO_DIR, "..", "logs")
CKPT_DIR = os.path.join(REPRO_DIR, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_ds, test_ds, used_real = load_mnist16(args.n_train, args.n_test, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SmallCNN()
    n_params = count_params(model)
    print(f"[classifier] params={n_params} used_real_mnist={used_real}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    logger = StepLogger(
        os.path.join(LOGS_DIR, "curves", "classifier_train.jsonl"),
        os.path.join(LOGS_DIR, "classifier_train.log"),
    )
    logger.log(event="start", n_params=n_params, used_real_mnist=used_real, args=vars(args))

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            if step % 10 == 0:
                acc = (logits.argmax(-1) == labels).float().mean().item()
                logger.log(event="step", epoch=epoch, step=step, loss=round(loss.item(), 4), train_acc=round(acc, 4))
            step += 1

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                logits = model(imgs)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.numel()
        test_acc = correct / total
        logger.log(event="epoch_end", epoch=epoch, test_acc=round(test_acc, 4), elapsed_sec=round(time.time() - t0, 1))

    elapsed = time.time() - t0
    logger.log(event="done", elapsed_sec=round(elapsed, 1), final_test_acc=round(test_acc, 4))

    ckpt_path = os.path.join(CKPT_DIR, "classifier.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[classifier] saved checkpoint to {ckpt_path}, wall_clock={elapsed:.1f}s, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
