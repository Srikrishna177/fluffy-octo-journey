"""Gaussian diffusion (DDPM, epsilon-prediction) with T=50 steps, and the pieces
DDPO needs: per-step reverse-transition log-probs and full trajectory sampling.

Beta schedule: linear, T=50 (spec allows linear or cosine; linear chosen for
simplicity — documented in RUN_LOG.md).
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
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]]
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # clip index-0 variance to index-1's value to avoid log(0) (standard DDPM trick)
        posterior_variance_clipped = posterior_variance.clone()
        posterior_variance_clipped[0] = posterior_variance[1]
        self.posterior_variance = posterior_variance_clipped

        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x0, t, noise):
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise

    def p_mean_variance(self, model, x_t, t):
        """t: [B] int64. Returns (mean [B,1,16,16], std [B] scalar-per-batch-elt broadcastable)."""
        eps_pred = model(x_t, t)
        sqrt_recip_ac = 1.0 / self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        pred_x0 = sqrt_recip_ac * x_t - (sqrt_one_minus_ac / self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)) * eps_pred
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        mean = coef1 * pred_x0 + coef2 * x_t
        var = self.posterior_variance[t]  # [B]
        std = torch.sqrt(var).view(-1, 1, 1, 1)
        return mean, std

    @staticmethod
    def gaussian_log_prob(x, mean, std):
        """Diagonal Gaussian log-density, summed over all non-batch dims -> [B]."""
        var = std ** 2
        log_prob = -0.5 * (((x - mean) ** 2) / var + torch.log(2 * math.pi * var))
        return log_prob.flatten(1).sum(dim=1)

    @torch.no_grad()
    def sample_trajectory(self, model, batch_size, img_size=16, device="cpu", generator=None):
        """Full ancestral sampling loop, t = T-1 .. 0. Returns a dict of stacked
        tensors (time-major, shape [T, B, ...]) needed for a DDPO update:
          x_t:       state BEFORE each step         [T,B,1,H,W]
          x_prev:    state AFTER each step           [T,B,1,H,W]
          timesteps: the t used at each step          [T]
          old_log_probs: log-prob of x_prev under the sampling-time policy [T,B]
          x0_final: final sample (== x_prev[-1])      [B,1,H,W]
        """
        x_t = torch.randn(batch_size, 1, img_size, img_size, device=device, generator=generator)
        xs, x_prevs, old_lps = [], [], []
        for t_int in reversed(range(self.T)):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            mean, std = self.p_mean_variance(model, x_t, t)
            noise = torch.randn(x_t.shape, device=device, generator=generator)
            x_prev = mean + std * noise
            lp = self.gaussian_log_prob(x_prev, mean, std)
            xs.append(x_t)
            x_prevs.append(x_prev)
            old_lps.append(lp)
            x_t = x_prev
        return {
            "x_t": torch.stack(xs, dim=0),
            "x_prev": torch.stack(x_prevs, dim=0),
            "timesteps": torch.tensor(list(reversed(range(self.T))), device=device),
            "old_log_probs": torch.stack(old_lps, dim=0),
            "x0_final": x_t,
        }
