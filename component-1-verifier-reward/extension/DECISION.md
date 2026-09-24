# Component 1 — extension design

## Recap of reproduction result

DDPO's core claim reproduced cleanly at toy scale: an unconditional MNIST-16x16 DDPM was
turned into a reliable digit-"7" generator via pure policy-gradient RL against a frozen,
non-differentiable external CNN classifier (reward 0.067→1.0, frac-argmax-is-7 9.4%→100%).
The run also reproduced DDPO's own documented failure mode: reward over-optimization
(sample diversity collapsed, KL-to-base drifted to ~53 nats and was still rising when the
run was stopped at the reward plateau). Full detail: `../logs/RUN_LOG.md`.

That reproduction uses an **external** verifier: a CNN trained independently, from scratch,
on raw pixels — it shares nothing with the generator. This sidesteps the proposal's actual
question, which is about **internal** verifiers: can a generative model's own understanding
(features it already has, or that it develops through generation) serve as the training
signal, rather than a bolted-on independent critic?

## The question this extension tests

**Does an internal verifier — one whose features are derived from the generator's own
learned representations rather than an independently-trained network — work as a DDPO
reward, and does it change the reward-hacking dynamics observed in the reproduction?**

This is a direct, small-scale test of the proposal's component-1 thesis ("can understanding
generalize as a training signal for generation") rather than a cosmetic variation on DDPO.

## Design

1. **Internal verifier**: a linear probe trained on **frozen intermediate features of the
   pretrained DDPM's own UNet** (extracted from the bottleneck activation when denoising a
   sample at a fixed near-zero timestep, i.e. reading out of the generator's own
   representation of "what did I just draw"), classifying digit identity on real MNIST-16x16.
   The probe is trained once, frozen, after DDPM pretraining and before DDPO — same protocol
   as the external classifier in the reproduction, so the only variable that changes is
   *whose features the verifier reads*, not how it's trained or used. This keeps the
   comparison to the reproduction apples-to-apples.
2. **Treatment run**: identical DDPO fine-tuning loop, same hyperparameters, same target class
   (digit 7), same time-box as the reproduction (capped at the reproduction's actual spend,
   ~1200s, to match rather than exceed it) — the only change is reward = internal-probe
   confidence instead of external-classifier confidence.
3. **Independent judge**: reward curves from two different reward *sources* aren't
   comparable on their own (a probe and a classifier can be miscalibrated differently), so the
   external classifier from the reproduction (frozen, untouched) is reused as a common,
   independent judge of both the baseline (external-reward) and treatment (internal-reward)
   post-RL samples. This is the number that actually answers "did generation quality
   improve," decoupled from whether either reward source is easy to hack.
4. **What "reward hacking differs" means here, operationalized**: compare, between baseline
   and treatment, (a) final reward-std (diversity collapse severity), (b) KL-to-base growth
   rate, (c) the *external-judge* score on treatment samples vs. the external-reward
   baseline's own final external-classifier score (since baseline's reward *is* the external
   classifier, this is a fair same-metric comparison for both arms).

## Hypothesis and honest expectation

No strong prior claimed either way — plausible outcomes, both informative:
- Internal-verifier reward could be **easier to hack** (probe reads coarser/lower-capacity
  features than a dedicated pixel-level CNN, so adversarial artifacts might satisfy it with
  less visually-real change) — a negative result for "internal verifiers are a free
  upgrade," but a real, useful finding about representation quality mattering.
- Or it could **resist hacking better** (features tied to the generator's own denoising
  process may be harder to game without actually producing a real "7", since the probe and
  generator share representation space) — a positive, thesis-supporting result.

Either outcome gets reported as-is; this is exactly the kind of ablation the program's
honesty rule exists for.

## Time-box

Probe training: a few minutes (linear probe on frozen features, cheap). DDPO treatment run:
capped at ~1200s wall-clock (matching, not exceeding, the reproduction's actual spend, since
the reproduction already showed reward saturates well before 1800s at this scale — no reason
to budget more for the treatment arm). If the internal-verifier reward doesn't move at all
in that budget, that's a logged negative result, not a reason to extend the budget by default.
