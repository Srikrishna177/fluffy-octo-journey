# Component 3 extension — on-policy training states vs. exposure bias

## What this tests

The reproduction (`../logs/RUN_LOG.md`) found that progressive distillation's
first halving round (32→16 DDIM steps) collapsed immediately: the 16-step
student's samples became unstructured noise, masked by a misleadingly-stable
classifier-confidence score caused by population mode collapse (64/64
fixed-noise samples predicted the same digit class). Diagnostics ruled out
learning rate and confirmed the frozen teacher itself samples fine at reduced
step counts with no retraining — the collapse is specific to the
distillation *training* procedure.

`DECISION.md` hypothesizes **exposure bias**: the paper's own Algorithm 2
trains the student on noisy states `z_t` drawn from the **data marginal**
(real MNIST `x0` + fresh noise), never from the model's own actual rollout
trajectory starting at pure noise — a train/test mismatch that lets gradient
descent find a cheap "predict one confident digit for nearly everything"
shortcut instead of the real per-condition composite mapping.

**This extension asks:** does training the student on **on-policy** states —
states actually visited along the teacher's own 32-step DDIM rollout from
pure noise, rather than the data marginal — fix the collapse?

## Design

Everything is held fixed to the reproduction's round-1 (32→16) run except
the *source* of the training input states `z_ti`:

- Same frozen teacher (`repro/checkpoints/ddpm_base_teacher.pt`, loaded
  read-only), same 32-step DDIM schedule, same halving to 16 steps.
- Same student architecture/init (`TinyUNet`, deep copy of the teacher's
  weights), same `batch_size=64`, `lr=2e-4`, `round_budget_sec=300`,
  `round_max_steps=4000`.
- Same regression-target computation (`teacher_two_step_target` /
  `invert_x_target`), imported unchanged from `repro/src/distill.py` — only
  the input states change.
- **On-policy replay pool**: before training, the teacher is rolled out from
  pure noise on its own 32-step DDIM schedule 4,096 times; the state actually
  visited at each of the 16 pair-start timesteps is recorded into a pool
  (16 pools of 4,096 states each). During training, minibatches draw `z_ti`
  from this pool instead of `q_sample(real_mnist_x0, t_i, fresh_eps)`.

Two 16-step students are compared under identical hyperparameters:
- **Baseline (data-marginal)**: reused directly from the reproduction —
  `repro/checkpoints/student_16step.pt` and its logged metrics.
- **Treatment (on-policy)**: trained here, `checkpoints/student_16step_onpolicy.pt`.

Evaluation uses the exact same metrics and shared fixed-starting-noise eval
batch (seed 12345, n_eval=64) as the reproduction: trajectory pixel MSE vs.
teacher, classifier-KL vs. teacher, own classifier-judge top-1
confidence/entropy, and the population-diversity metrics
(`population_class_entropy_nats`, `frac_samples_in_majority_class`) added
during the reproduction's diagnosis.

## Rerun

```bash
cd component-3-adaptive-distill/extension/src
python3 distill_onpolicy.py \
  --pool_size 4096 --pool_batch 512 --pool_seed 999 \
  --batch_size 64 --lr 2e-4 --round_budget_sec 300 --round_max_steps 4000 --train_seed 101 \
  --n_eval 64 --eval_seed 12345
python3 plots_ext.py
```

Requires the reproduction to have been run first (reads
`../../repro/checkpoints/ddpm_base_teacher.pt`, `classifier.pt`, and
`student_16step.pt`, all read-only). Outputs land in `logs/results.json`,
`logs/curves/distill_train_onpolicy.jsonl`, `logs/plots/*.png`,
`checkpoints/student_16step_onpolicy.pt`, `checkpoints/eval_samples.pt`.
CPU-only; no GPU required. Wall-clock and full result numbers:
`RUN_LOG.md`.

## Layout

- `src/distill_onpolicy.py` — pool generation + on-policy training round +
  evaluation of all three arms (teacher / data-marginal / on-policy).
- `src/plots_ext.py` — training-curve and sample-grid plots (read-only).
- `logs/results.json` — all three arms' metrics side by side.
- `logs/curves/distill_train_onpolicy.jsonl` — per-step training loss.
- `logs/plots/onpolicy_training_curve.png`,
  `logs/plots/sample_grid_comparison.png`.
- `checkpoints/student_16step_onpolicy.pt`, `checkpoints/eval_samples.pt`.

## RunPod full-scale rerun

Same scaffold as the reproduction (`../repro/Dockerfile`,
`../repro/requirements.txt`, `../repro/runpod_launch.sh`) — the on-policy
pool-generation + training mechanism here is a drop-in replacement for
`repro/src/distill.py`'s data-marginal sampling and generalizes directly to
the full-scale model/data/schedule described there.
