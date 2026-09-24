# DDPO toy-scale reproduction

Faithful small-scale reproduction of DDPO's core claim (Black et al., "Training
Diffusion Models with Reinforcement Learning," arXiv 2305.13301): treat the
reverse diffusion sampling chain as a multi-step MDP and fine-tune a pretrained
diffusion model against a black-box scalar reward via policy-gradient RL,
without backpropagating through the reward.

Design spec: `component-1-verifier-reward/DECISION.md`. Results and honest
write-up: `component-1-verifier-reward/logs/RUN_LOG.md`.

## Setup

```bash
cd component-1-verifier-reward/repro
pip install -r requirements.txt   # torch, torchvision, numpy, matplotlib (CPU is fine)
```

## Run (in order)

All scripts write logs to `../logs/` (curves as JSONL + a human-readable
`.log`, checkpoints to `checkpoints/`).

```bash
cd src

# 1. Pretrain the frozen verifier/reward model: small CNN on MNIST-16x16.
#    Cheap (~10s on 4 CPU cores). Saves checkpoints/classifier.pt.
python3 train_classifier.py --epochs 8 --n_train 20000 --n_test 5000 --batch_size 128

# 2. Pretrain the base DDPM (TinyUNet, epsilon-prediction, T=50) from scratch,
#    unconditional. Time-boxed; --budget_sec caps wall clock (default here:
#    1800s = 30 min). Saves checkpoints/ddpm_base.pt.
python3 train_ddpm.py --budget_sec 1800 --n_train 20000 --n_test 5000 --batch_size 128

# 3. DDPO RL fine-tuning: policy-gradient fine-tune the base DDPM against the
#    frozen classifier's confidence for a fixed target digit (default: "7").
#    Time-boxed the same way. Saves checkpoints/ddpo_finetuned.pt.
python3 train_ddpo.py --budget_sec 1800 --batch_size 48 --target_class 7

# 4. Eval: sample a fixed held-out batch of 64 from the base and fine-tuned
#    models (same noise seed for both), score with the classifier, save
#    before/after sample grids and all training-curve plots.
python3 eval.py --n_eval 64 --target_class 7
```

## Results at a glance

See `component-1-verifier-reward/logs/RUN_LOG.md` for the full narration,
exact hyperparameters and wall-clock times actually used, and
`component-1-verifier-reward/logs/eval_before_after.json` for the headline
before/after reward numbers. Plots (loss curves, DDPO reward-over-iterations,
KL-to-base, before/after sample grids) are in
`component-1-verifier-reward/logs/plots/`.

## What each file implements

- `src/data.py` — MNIST loading + downsampling to 16x16, scaled to [-1,1].
  Falls back to a documented synthetic substitute if the real download is
  blocked (not needed in this run — real MNIST downloaded successfully).
- `src/classifier.py` — the frozen reward model (small CNN) and `reward_fn`.
- `src/unet.py` — `TinyUNet`, the epsilon-prediction DDPM backbone (~300k
  params).
- `src/diffusion.py` — beta schedule, `q_sample`, the reverse-transition
  mean/variance (`p_mean_variance`), the closed-form diagonal-Gaussian
  log-prob, and `sample_trajectory` (the full T-step ancestral sampler that
  DDPO needs — it returns every `(x_t, x_{t-1})` pair and the sampling-time
  log-prob of each transition).
- `src/train_ddpm.py` — standard DDPM pretraining (MSE on predicted noise).
- `src/train_ddpo.py` — DDPO Algorithm 1: sample trajectories, score the final
  image with the frozen reward model, standardize reward across the batch as
  the advantage (broadcast to every step, since the reward is terminal),
  recompute each transition's log-prob under the current policy, and take a
  REINFORCE / PPO-clipped policy-gradient step. Also estimates KL-to-base
  (closed form, since both models share the same fixed step variance).
- `src/eval.py` — before/after eval on a fixed seed, sample grids, curve
  plots.

## Full-scale (RunPod) path

`Dockerfile`, `requirements-fullscale.txt`, and `runpod_launch.sh` scaffold
rerunning the same DDPO mechanism at full paper scale (T=1000, a real
diffusers UNet, CIFAR/SD-scale data, ImageReward as the reward model) on an
A100 pod. See the comments at the top of `runpod_launch.sh` for exactly what
needs to be swapped in versus this toy run. Not built or executed in this
sandbox.
