"""Train the bidirectional transformer with MaskGIT's masked-token cross-entropy
objective and cosine masking-ratio schedule. Time-boxed by --budget_sec.

Writes:
  - checkpoints/maskgit.pt (final/last-saved model state)
  - ../logs/curves/maskgit_train.jsonl (loss every N steps)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from data import load_mnist16_tokens, MASK_TOKEN, SEQ_LEN
from model import BidirectionalMaskedTransformer, count_params, sample_training_mask

HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, "..", "checkpoints")
LOG_DIR = os.path.join(HERE, "..", "..", "logs")
CURVE_DIR = os.path.join(LOG_DIR, "curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_sec", type=float, default=1800.0)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every_sec", type=float, default=120.0)
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    t_data0 = time.time()
    train_tokens, train_labels, test_tokens, test_labels, used_real = load_mnist16_tokens(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed
    )
    print(f"[train] data loaded in {time.time()-t_data0:.1f}s, used_real_mnist={used_real}, "
          f"train={train_tokens.shape}, test={test_tokens.shape}", flush=True)

    model = BidirectionalMaskedTransformer().to(device)
    n_params = count_params(model)
    print(f"[train] model params: {n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(reduction="none")

    n_train = train_tokens.shape[0]
    curve_path = os.path.join(CURVE_DIR, "maskgit_train.jsonl")
    curve_f = open(curve_path, "w")

    t0 = time.time()
    last_ckpt_t = t0
    step = 0
    rolling_loss = []
    perm = torch.randperm(n_train)
    ptr = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= args.budget_sec:
            print(f"[train] budget exhausted at step {step}, elapsed={elapsed:.1f}s", flush=True)
            break

        if ptr + args.batch_size > n_train:
            perm = torch.randperm(n_train)
            ptr = 0
        idx = perm[ptr: ptr + args.batch_size]
        ptr += args.batch_size
        batch = train_tokens[idx].to(device)  # [B,256] in {0,1,2,3}

        mask = sample_training_mask(batch.shape[0], SEQ_LEN, device)  # True=masked
        inp = batch.clone()
        inp[mask] = MASK_TOKEN

        logits = model(inp)  # [B,256,4]
        loss_all = ce(logits.reshape(-1, logits.shape[-1]), batch.reshape(-1))  # [B*256]
        loss_all = loss_all.view(batch.shape)
        denom = mask.sum().clamp(min=1)
        loss = (loss_all * mask).sum() / denom

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        rolling_loss.append(loss.item())
        if len(rolling_loss) > 100:
            rolling_loss.pop(0)

        if step % args.log_every == 0:
            avg_loss = sum(rolling_loss) / len(rolling_loss)
            with torch.no_grad():
                acc = (logits.argmax(-1) == batch)[mask].float().mean().item()
            rec = {"step": step, "elapsed_sec": round(elapsed, 2), "loss": loss.item(),
                   "avg_loss_100": avg_loss, "masked_token_acc": acc}
            curve_f.write(json.dumps(rec) + "\n")
            curve_f.flush()
            print(f"[train] step={step} elapsed={elapsed:.1f}s loss={loss.item():.4f} "
                  f"avg100={avg_loss:.4f} masked_acc={acc:.4f}", flush=True)

        if time.time() - last_ckpt_t >= args.ckpt_every_sec:
            torch.save({"model": model.state_dict(), "step": step, "n_params": n_params},
                       os.path.join(CKPT_DIR, "maskgit.pt"))
            last_ckpt_t = time.time()

        step += 1

    total_elapsed = time.time() - t0
    torch.save({"model": model.state_dict(), "step": step, "n_params": n_params,
                "total_elapsed_sec": total_elapsed, "used_real_mnist": used_real},
               os.path.join(CKPT_DIR, "maskgit.pt"))
    curve_f.close()
    print(f"[train] DONE. steps={step} total_elapsed={total_elapsed:.1f}s "
          f"checkpoint=checkpoints/maskgit.pt", flush=True)


if __name__ == "__main__":
    main()
