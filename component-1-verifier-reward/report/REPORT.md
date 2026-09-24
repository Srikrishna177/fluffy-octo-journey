# Component 1: Verifier reward for diffusion generation — DDPO reproduction + internal-verifier extension

## The paper reproduced

**DDPO** (Black, Janner, Du, Kostrikov, Levine, "Training Diffusion Models
with Reinforcement Learning," ICLR 2024, arXiv:2305.13301). Its core claim:
treat the reverse diffusion sampling chain as a multi-step MDP and fine-tune
a *pretrained* diffusion model against an arbitrary, non-differentiable
scalar reward via policy-gradient RL (a PPO-style surrogate over the
per-step Gaussian reverse-transition log-probabilities), without ever
backpropagating through the reward model itself.

## The small-scale reproduction: what it found

Toy setup (CPU-only, 4 cores, 15GB RAM, no GPU): MNIST downsampled to
16x16, a ~309k-parameter TinyUNet DDPM (T=50 denoising steps, vs. 1000 in
the paper) pretrained from scratch, and a frozen ~72k-parameter external
CNN classifier as the reward model (softmax confidence for digit "7").
Full design: `../DECISION.md`; full log: `../logs/RUN_LOG.md`.

**Result — DDPO's core claim reproduced cleanly at this scale.** On a fixed
64-sample held-out eval batch (same noise seed before/after):

| | before RL | after RL |
|---|---|---|
| mean reward (classifier P(class=7)) | 0.0672 | **0.9998** |
| fraction classified as "7" | 9.4% | **100%** |

Pure policy-gradient RL against a frozen, black-box reward turned an
unconditional generator into a reliable "7" generator, with no gradient
ever flowing through the classifier.

**The reproduction also honestly reproduced DDPO's own documented failure
mode: reward over-optimization.** Once reward saturated (~iteration 185,
~915s wall clock), the run was kept going a further ~280s (to ~1195s,
matching the operator's decision to stop at a sustained plateau rather than
the full 1800s budget) and showed: sample diversity collapsing (reward std
0.1766 -> 0.0006), a fine-grained noise texture appearing on top of
otherwise genuine "7"-shaped digits, and KL-to-base drifting monotonically
from ~1.8 nats to **~53.4 nats**, still rising when the run was stopped.

## The extension: internal vs. external verifier

The reproduction's reward is an **external** verifier — a CNN trained
independently, from scratch, on raw pixels, sharing nothing with the
generator. This sidesteps the actual proposal question: can a generator's
*own* understanding serve as its training signal?

**Question tested:** does an **internal** verifier — a linear probe on the
generator's own frozen intermediate features — work as a DDPO reward, and
does it change the reward-hacking dynamics? Design: `../extension/DECISION.md`.

**Design, holding everything constant except the reward source:**
1. A linear probe (40,970 params) trained on the frozen pretrained DDPM's
   own bottleneck activation (the `mid` ResBlock, read via a forward hook,
   at a fixed near-zero timestep t=0), predicting MNIST digit identity.
   Test accuracy: **97.62%** (vs. 98.02% for the external CNN) — the DDPM's
   own denoising features are almost as linearly separable by digit
   identity as a purpose-trained classifier, despite never being trained
   for classification.
2. An identical DDPO run (same TinyUNet policy init, same T=50,
   batch_size=48, lr=1e-4, num_inner_epochs=1, clip_eps=1e-4, target class
   7), with the reward swapped to the frozen probe's confidence.
3. Independent-judge eval: the treatment's post-RL samples scored by the
   **original, untouched external classifier** — never used in this arm's
   training — on the same 64-sample / seed-12345 protocol as the
   reproduction, so the numbers are directly comparable.

## The real result

Full log with every number's provenance: `../extension/RUN_LOG.md`. Both
runs' raw JSONL/JSON logs are the source of truth; nothing below is
projected.

| metric | reproduction (external reward) | extension (internal reward) |
|---|---|---|
| final training reward (own reward source) | 0.9999 | 0.9999 |
| final training reward-std (own reward source) | 0.0003 | 0.0001 |
| iterations / wall clock to stop | 244 / 1195.4s | 115 / 505.5s |
| wall clock to first sustained saturation | ~915s | ~400s |
| external-judge reward_mean (post-RL) | 0.9998 | 0.9594 |
| external-judge reward_std (post-RL) | 0.0006 | 0.1268 |
| external-judge frac. argmax==7 (post-RL) | 100% | 98.4% |
| KL-to-base trajectory | monotonic climb to **53.4 nats**, still rising at stop | rises to a peak of ~18.8 nats then **plateaus/declines to ~10.3 nats** |

Both time-boxed the same way: the extension was stopped once its own
reward had been pinned at the ceiling (std < 0.002) for 20+ consecutive
iterations, exactly the criterion the reproduction used at ~1195s.

**The number that matters most isn't in the table.** The sample grids
(`../extension/logs/plots/samples_after_rl.png` vs. the reproduction's
`../logs/plots/samples_after_rl.png`) show a qualitative difference the
scalar metrics obscure: the reproduction's post-RL samples are every one
recognizably, unambiguously "7"-shaped (top bar + diagonal downstroke),
just low-diversity and textured. The extension's post-RL samples instead
collapse onto a **cruder pattern** — a bright horizontal bar with faint
scattered noise below, with no consistent diagonal stroke — that is far
less convincing as a handwritten "7" to a human eye, even though it fools
the internal probe almost perfectly and fools the independent external
classifier reasonably well too (95.9% mean confidence, 98.4% argmax rate).

## Which hypothesis this supports

`extension/DECISION.md` posed two candidate outcomes with no strong prior:
internal-verifier reward could be easier to hack (coarser probe features
let the policy get away with a less-real solution) or harder to hack
(features tied to the generator's own representations resist gaming).

**The result is genuinely mixed, and we report it as such rather than
forcing a clean story:**

- **The qualitative evidence (sample grids) supports "easier to hack."**
  The internal-reward run converged to a visually cruder, less
  recognizably-"7" shortcut than the baseline's genuine (if
  texture-corrupted) 7-shapes — exactly what a coarser, lower-capacity
  4096-dim linear probe would be expected to allow.
- **The numeric optimization dynamics (KL plateauing instead of climbing)
  superficially look like "harder to hack," but we judge this to most
  likely be a confound, not a real robustness difference.** Once the
  internal-reward policy found its horizontal-bar shortcut, the probe's
  confidence was already pinned near 1.0 for essentially every sample in
  the batch, leaving very little residual reward variance for the
  standardized-advantage objective to amplify into further gradient
  pressure — so the policy simply stopped moving, rather than being
  actively resistant to further gaming. The baseline's external classifier,
  by contrast, kept offering a thin residual gradient (confidence
  improving in the fourth decimal place) that kept getting amplified into
  continued drift long after the reward "looked" saturated.
- **An honest limitation we did not control for:** the extension's
  post-saturation window (~105s at the ceiling) is much shorter than the
  baseline's (~280s), because it was stopped as soon as it hit a
  comparably sustained plateau. We cannot rule out that letting it run for
  the same amount of time *at* saturation would eventually show the same
  continued KL creep. A follow-up should match wall-clock time spent at
  saturation, not total wall-clock, between arms.

**Bottom line:** on the visual/semantic evidence, this small-scale result
leans toward "internal-verifier reward is easier to hack" — but the
numeric KL/diversity story is ambiguous and likely reflects how much
residual gradient signal was left after each reward source's own
saturation point, not a genuine difference in robustness to gaming.

## What this does and doesn't show at this scale

**Does show:** (1) DDPO's core RL-from-black-box-reward mechanism works
end-to-end at toy scale for both an external and an internal reward
source; (2) a linear probe on a diffusion model's own frozen intermediate
features is almost as good a digit classifier as a purpose-trained CNN
(97.6% vs. 98.0%), i.e. generative pretraining does produce linearly
decodable class-relevant features, supporting the premise that "internal
understanding" is available to exploit; (3) at this toy scale, DDPO
fine-tuning against that internal signal reward-hacks it into a cruder,
less semantically faithful degenerate solution than the same procedure
does against an external CNN.

**Doesn't show:** anything about whether this pattern holds at the scale
the actual proposal targets — a unified multimodal model with billions of
parameters, richer image statistics than 16x16 binary-ish MNIST digits, and
an internal verifier that is far higher-capacity than a single linear
layer on a 4096-dim bottleneck. A 4096-dim linear probe is a much weaker
"understanding" module than what a real unified model's own decoder
features would offer; the easier-to-hack result here may be specific to
that capacity gap rather than to internal-vs-external verifiers in
general. It also doesn't control for time-at-saturation between arms (see
limitation above), so the KL-drift comparison should be read as suggestive,
not conclusive.

**Path to full-scale validation:** `../repro/Dockerfile`,
`../repro/requirements-fullscale.txt`, and `../repro/runpod_launch.sh`
scaffold rerunning the identical DDPO mechanism on an A100 pod (T=1000, a
real diffusers UNet, CIFAR/SD-scale data, ImageReward as the external
reward). The same scaffold applies unchanged to the extension — the only
addition needed is a probe-training stage reading a real diffusion
backbone's intermediate features (much higher-dimensional and more
expressive than TinyUNet's 4096-dim bottleneck) before swapping it in as
the DDPO reward, which would directly test whether the capacity gap
diagnosed above is in fact what drove the "easier to hack" result here.
Not built or executed in this CPU sandbox; see `../extension/README.md` for
how to rerun the toy version end to end.
