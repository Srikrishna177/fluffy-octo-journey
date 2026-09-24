"""Deterministic DDIM sampler + progressive-distillation schedule utilities on
top of Component 1's frozen TinyUNet DDPM.

Noise schedule (T=50, linear beta 1e-4 -> 0.02) is copied EXACTLY from
component-1's diffusion.py, since the frozen ddpm_base.pt checkpoint was
trained against that specific schedule -- DDIM sampling reuses the same
epsilon-predictor and the same alphas_cumprod, only the *sampling procedure*
changes from ancestral (stochastic) to DDIM (deterministic, eta=0).
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class DiffusionConfig:
    T: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02


class GaussianDiffusion:
    def __init__(self, cfg: DiffusionConfig, device="cpu"):
        self.T = cfg.T
        self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        # DDIM/progressive-distillation notation: alpha_t = sqrt(abar_t), sigma_t = sqrt(1-abar_t)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise):
        """Same forward-process sample used for DDPM training: z_t = alpha_t*x0 + sigma_t*eps.
        t: [B] int64 indices into the T=50 grid."""
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise

    def alpha_sigma(self, t):
        """t: [B] int64 or python int index into the T-grid. Returns (alpha_t, sigma_t)."""
        return self.sqrt_alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]

    # ---- DDIM (eta=0, deterministic) single-step update ----
    def ddim_step(self, model, x_t, t_cur: int, t_next):
        """One deterministic DDIM step from timestep t_cur to t_next.
        t_next may be None, meaning 'jump straight to x0' (alpha=1, sigma=0).
        Returns (x_next, eps_pred, pred_x0)."""
        b = x_t.shape[0]
        t_cur_batch = torch.full((b,), t_cur, device=x_t.device, dtype=torch.long)
        eps_pred = model(x_t, t_cur_batch)
        alpha_cur = self.sqrt_alphas_cumprod[t_cur]
        sigma_cur = self.sqrt_one_minus_alphas_cumprod[t_cur]
        pred_x0 = (x_t - sigma_cur * eps_pred) / alpha_cur
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        if t_next is None:
            x_next = pred_x0
        else:
            alpha_next = self.sqrt_alphas_cumprod[t_next]
            sigma_next = self.sqrt_one_minus_alphas_cumprod[t_next]
            x_next = alpha_next * pred_x0 + sigma_next * eps_pred
        return x_next, eps_pred, pred_x0

    @torch.no_grad()
    def ddim_sample(self, model, schedule, batch_size, img_size=16, device="cpu", generator=None):
        """Full deterministic DDIM sampling loop over `schedule` (a list of int
        timesteps, descending, e.g. [49, 46, ..., 0]). NFE == len(schedule).
        Returns the final clean sample [B,1,H,W]."""
        x = torch.randn(batch_size, 1, img_size, img_size, device=device, generator=generator)
        for i, t_cur in enumerate(schedule):
            t_next = schedule[i + 1] if i + 1 < len(schedule) else None
            x, _, _ = self.ddim_step(model, x, t_cur, t_next)
        return x


def make_ddim_schedule(T: int, n_steps: int):
    """Evenly-spaced descending subsequence of {0,...,T-1} of length n_steps,
    e.g. T=50, n_steps=32 -> 32 unique descending ints spanning [T-1, 0]."""
    idx = torch.linspace(T - 1, 0, n_steps).round().long()
    sched = idx.tolist()
    # dedupe while preserving order (guards against rounding collisions)
    out = []
    for v in sched:
        if not out or out[-1] != v:
            out.append(v)
    return out


def halve_schedule(schedule):
    """Take every other timestep -> exactly half the steps, nested inside the
    original schedule (required so 'two old steps == one new step' is exact)."""
    assert len(schedule) % 2 == 0, f"schedule length {len(schedule)} must be even to halve"
    return schedule[0::2]
