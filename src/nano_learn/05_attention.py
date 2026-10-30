"""
Chapter 05 — Attention (the core of transformers)

Intuition:
  Each token builds Query (what am I looking for?), Key (what do I contain?),
  and Value (what do I pass along?). Attention weights = softmax(Q K^T / √d).
  Output = weighted sum of Values.

  Scaling by √d keeps dot products from growing with dimension (which would
  push softmax into tiny-gradient extremes).

Run:
  uv run python src/nano_learn/05_attention.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V: (batch, heads, seq, head_dim) or (batch, seq, dim)
    Returns (output, weights).
    """
    d = Q.shape[-1]
    # scores[b, i, j] ≈ how much token i attends to token j
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)

    if causal:
        # Future tokens are blocked: language models only look left.
        seq = scores.size(-1)
        mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    return out, weights


class SelfAttention(nn.Module):
    """Single-head self-attention for teaching (multi-head is several of these in parallel)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, dim)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        return scaled_dot_product_attention(Q, K, V, causal=causal)


def main() -> None:
    torch.manual_seed(0)
    batch, seq, dim = 1, 5, 8
    x = torch.randn(batch, seq, dim)

    attn = SelfAttention(dim)
    out, weights = attn(x, causal=True)

    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}  (same as input — each token gets a new vector)")
    print("\nCausal attention weights (rows = query position, cols = key position):")
    print(weights[0].detach().numpy().round(3))
    print("\nNote the upper triangle is ~0: token i cannot attend to future token j > i.")
    print("Next: 06_mini_gpt.py + scripts/train_mini_gpt.py (stack attention into a tiny LM).")


if __name__ == "__main__":
    main()
