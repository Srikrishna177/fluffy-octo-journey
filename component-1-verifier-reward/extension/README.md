# Component 1 extension — internal vs. external verifier reward

## What this tests

The reproduction (`../repro/`) turned a pretrained toy MNIST DDPM into a
reliable digit-"7" generator via DDPO policy-gradient RL against a frozen
**external** CNN classifier (trained independently, from scratch, on raw
pixels — it shares nothing with the generator).

This extension asks: does an **internal** verifier — a linear probe reading
the generator's own frozen intermediate features — work as a DDPO reward,
and does it change the reward-hacking dynamics (diversity collapse, KL
drift) observed in the reproduction? Full design rationale: `DECISION.md`.
Full results and numeric comparison: `RUN_LOG.md`.

Everything is held constant versus the reproduction except the reward
source: same frozen pretrained DDPM (`../repro/checkpoints/ddpm_base.pt`,
never retrained), same DDPO Algorithm 1 implementation, same hyperparameters
(T=50, batch_size=48, lr=1e-4, num_inner_epochs=1, clip_eps=1e-4, target
class 7), same ~1200s time-box.

## Files

- `src/probe.py` — `FeatureExtractor` (forward hook on the pretrained
  `TinyUNet`'s `mid` bottleneck ResBlock, run at a fixed near-zero timestep
  t=0) and `LinearProbe` (a single linear layer on top). Reuses
  `repro/src/unet.py`'s `TinyUNet` unmodified — the hook is attached from
  outside, no changes to the model class.
- `src/train_probe.py` — trains the linear probe (only the probe; the UNet
  stays frozen throughout) on real MNIST-16x16 labels, using
  `repro/src/data.py`'s data pipeline.
- `src/train_ddpo_internal.py` — the treatment run: identical to
  `repro/src/train_ddpo.py` except `reward_fn` (external classifier) is
  replaced by `probe.internal_reward_fn` (frozen base-UNet features -> frozen
  linear probe -> softmax P(class=7)).
- `src/eval_internal.py` — independent-judge eval: same protocol as
  `repro/src/eval.py` (n=64, seed=12345), but scores the treatment's
  post-RL samples with the **original frozen external classifier**
  (`repro/checkpoints/classifier.pt`), so the number is directly comparable
  to the reproduction's own `eval_before_after.json`.

## Rerun

```bash
cd component-1-verifier-reward/extension/src

# 1. Train the internal-verifier linear probe (frozen DDPM features -> digit label).
python3 train_probe.py --epochs 8 --n_train 20000 --n_test 5000 --batch_size 128 --lr 1e-3 --seed 0

# 2. DDPO treatment run: reward = internal-probe confidence, everything else identical
#    to repro/src/train_ddpo.py. Time-boxed to ~1200s to match the reproduction's spend.
python3 train_ddpo_internal.py --budget_sec 1200 --batch_size 48 --lr 1e-4 --T 50 \
  --target_class 7 --num_inner_epochs 1 --clip_eps 1e-4 --seed 1 --ckpt_every_sec 120 --kl_every 5

# 3. Independent-judge eval + plots.
python3 eval_internal.py --n_eval 64 --target_class 7 --seed 12345
```

Logs land in `logs/` (curves as JSONL + `.log`, plots in `logs/plots/`,
checkpoints in `checkpoints/`), mirroring `repro/`'s layout.

## Full-scale (RunPod) path

No separate Dockerfile/launch script is duplicated here: the extension reuses
the exact same model/data/dependency stack as the reproduction, so
`../repro/Dockerfile` and `../repro/runpod_launch.sh` apply unchanged — the
only full-scale addition needed is a fourth stage calling
`train_probe.py` + `train_ddpo_internal.py` in place of (or alongside)
`train_ddpo.py`, with the same T=1000 / diffusers-UNet / CIFAR-or-SD-scale
swaps documented in `runpod_launch.sh`'s header comment. Not built or
executed in this CPU sandbox.
