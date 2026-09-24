"""Internal verifier: a linear probe on frozen intermediate features of the
already-pretrained DDPM's own UNet (repro/checkpoints/ddpm_base.pt).

We do NOT modify or retrain the UNet. We reuse repro/src/unet.py's TinyUNet
class as-is and attach a forward hook to its `mid` ResBlock (the bottleneck,
8x8x64 = 4096-dim activation map) to read out its intermediate features
during a normal forward pass at a fixed near-zero timestep (t=0). A single
linear layer is then trained on TOP of those frozen features to classify
MNIST digit identity. Only the linear layer's weights are ever updated here;
the UNet is used strictly in eval()/no_grad() mode as a frozen feature
extractor, exactly mirroring how the external classifier is used as a frozen
scorer in the reproduction.
"""
import torch
import torch.nn as nn


FEATURE_TIMESTEP = 0  # "near-zero" timestep at which we read the bottleneck feature
MID_FEATURE_DIM = 64 * 8 * 8  # TinyUNet(base_ch=32).mid output shape: [B,64,8,8]


class FeatureExtractor:
    """Wraps a frozen TinyUNet with a forward hook on `mid`, exposing its
    bottleneck activation for the most recent forward() call. The UNet
    itself is never modified (no new methods added to unet.py) and never
    updated (all extraction happens under torch.no_grad())."""

    def __init__(self, unet: nn.Module):
        self.unet = unet
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self._feat = None
        self._handle = self.unet.mid.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out

    @torch.no_grad()
    def extract(self, x0_or_xt: torch.Tensor, t_value: int = FEATURE_TIMESTEP) -> torch.Tensor:
        """x0_or_xt: [B,1,16,16] image (roughly in [-1,1]).
        Runs the frozen UNet forward at timestep t_value and returns the
        flattened bottleneck feature [B, MID_FEATURE_DIM]."""
        b = x0_or_xt.shape[0]
        t = torch.full((b,), t_value, dtype=torch.long, device=x0_or_xt.device)
        _ = self.unet(x0_or_xt, t)
        feat = self._feat
        return feat.flatten(1).clone()

    def remove(self):
        self._handle.remove()


class LinearProbe(nn.Module):
    """A single linear layer on top of frozen UNet bottleneck features."""

    def __init__(self, feature_dim: int = MID_FEATURE_DIM, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


def internal_reward_fn(extractor: FeatureExtractor, probe: LinearProbe,
                        images: torch.Tensor, target_class: int) -> torch.Tensor:
    """images: [B,1,16,16]. Returns [B] softmax confidence for target_class,
    computed from the FROZEN base UNet's bottleneck feature + the FROZEN
    trained linear probe head — the internal-verifier analogue of
    classifier.reward_fn. No gradient flows through this function (matches
    the external-classifier reward: reward is a frozen, non-differentiable
    scalar; only the policy's own log-probs carry gradient in DDPO)."""
    with torch.no_grad():
        feat = extractor.extract(images, FEATURE_TIMESTEP)
        logits = probe(feat)
        probs = torch.softmax(logits, dim=-1)
    return probs[:, target_class]
