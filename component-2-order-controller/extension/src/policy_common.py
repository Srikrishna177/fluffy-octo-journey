"""Shared feature-extraction + policy-network code, used by both the
training-data generator (gen_policy_data.py) and the 4th-arm inference
script (inference_ext.py), so the exact same feature definition is used at
train time and eval time.

Feature vector per masked position (documented choice, extension/DECISION.md
leaves this open: "hidden state / logit-derived features plus a positional
embedding/index"):
  - hidden_state: the frozen transformer's own d_model=96 pre-head hidden
    vector at that position (post encoder + final LayerNorm, i.e. exactly the
    vector the model's own `head` linear layer sees before producing logits).
  - normalized position index: position / (seq_len - 1), in [0, 1].
  - max softmax confidence: logit-derived, the same raw signal the
    reproduction's "confidence" arm used alone -- given to the policy as ONE
    input among several, not the only one.
  - predictive entropy: logit-derived, an additional uncertainty signal.

Total feature dim = 96 + 1 + 1 + 1 = 99.

No weights of the frozen MaskGIT transformer are modified anywhere here --
`forward_with_hidden` only calls its existing submodules (tok_emb, pos_emb,
encoder, norm_out, head), it does not add new parameters to that model.
"""
import torch
import torch.nn as nn

FEATURE_DIM = 99
FEATURE_SPEC = (
    "hidden_state(96, post-encoder+LayerNorm, pre-head) + "
    "normalized_position_index(1) + max_softmax_confidence(1) + predictive_entropy(1)"
)


def forward_with_hidden(model, tokens: torch.Tensor):
    """Replicates BidirectionalMaskedTransformer.forward but also returns the
    pre-head hidden state. Calls the model's own submodules only -- no new
    parameters, no modification of its weights.
    tokens: LongTensor [B, L]. Returns (hidden [B,L,d_model], logits [B,L,V]).
    """
    B, L = tokens.shape
    pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
    x = model.tok_emb(tokens) + model.pos_emb(pos_ids)
    x = model.encoder(x)
    x = model.norm_out(x)
    logits = model.head(x)
    return x, logits


def extract_features(hidden: torch.Tensor, logits: torch.Tensor, seq_len: int):
    """hidden: [B,L,D], logits: [B,L,V]. Returns (features [B,L,FEATURE_DIM], pred [B,L])."""
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)  # [B,L]
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)  # [B,L]
    B, L, D = hidden.shape
    pos_idx = torch.arange(L, device=hidden.device, dtype=torch.float32)
    pos_idx = (pos_idx / max(seq_len - 1, 1)).unsqueeze(0).expand(B, L)
    feat = torch.cat(
        [hidden, pos_idx.unsqueeze(-1), conf.unsqueeze(-1), entropy.unsqueeze(-1)], dim=-1
    )
    assert feat.shape[-1] == FEATURE_DIM
    return feat, pred


class PolicyMLP(nn.Module):
    """Small 2-hidden-layer MLP: predicts logit for P(argmax prediction at this
    position is correct) from the feature vector above."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logit, apply sigmoid outside
