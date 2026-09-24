"""Small bidirectional transformer for MaskGIT-style masked-token prediction.

Spec (DECISION.md): ~4 layers, small width, a few hundred K params. Token +
positional embeddings over the 256-position sequence (16x16 MNIST tokens,
vocab {0,1,2,3} for gray levels + MASK token id 4). Trained with standard
masked-token cross-entropy (predict only masked positions).
"""
import math

import torch
import torch.nn as nn

from data import VOCAB_SIZE, MASK_TOKEN, SEQ_LEN

TOTAL_VOCAB = VOCAB_SIZE + 1  # 4 gray levels + 1 mask token, as an input embedding vocab


class BidirectionalMaskedTransformer(nn.Module):
    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 192, seq_len: int = SEQ_LEN, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(TOTAL_VOCAB, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        # only predict the 4 real gray levels -- MASK is never a prediction target
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: LongTensor [B, seq_len] with values in {0,1,2,3,MASK_TOKEN}.
        Returns logits [B, seq_len, VOCAB_SIZE] (bidirectional, no causal mask:
        every position attends to every other position, as in MaskGIT/BERT).
        """
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(tokens) + self.pos_emb(pos_ids)
        x = self.encoder(x)
        x = self.norm_out(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cosine_mask_ratio(r: torch.Tensor) -> torch.Tensor:
    """MaskGIT's cosine masking-ratio schedule: gamma(r) = cos(r * pi/2), r in [0,1].
    r=0 -> ratio 1 (fully masked), r=1 -> ratio 0 (fully revealed)."""
    return torch.cos(r * math.pi / 2.0)


def sample_training_mask(batch_size: int, seq_len: int, device) -> torch.Tensor:
    """Returns a BoolTensor [B, seq_len], True = masked, with a per-example masking
    ratio drawn from the cosine schedule (r ~ Uniform(0,1))."""
    r = torch.rand(batch_size, device=device)
    ratio = cosine_mask_ratio(r)  # [B]
    n_mask = torch.clamp((ratio * seq_len).long(), min=1, max=seq_len)  # [B]
    # random permutation of positions per example; take the first n_mask[i] as masked
    rand_scores = torch.rand(batch_size, seq_len, device=device)
    thresh_idx = n_mask.unsqueeze(1) - 1  # [B,1]
    sorted_scores, _ = torch.sort(rand_scores, dim=1)
    thresh = torch.gather(sorted_scores, 1, thresh_idx.clamp(min=0))  # [B,1]
    mask = rand_scores <= thresh
    return mask
