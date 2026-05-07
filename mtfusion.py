"""
Shared building blocks for multi-token tabular + signal/image fusion (MTFusion family).

Used by ``mtfusion_resnet_models`` and ``waveform_fusion_models``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedder(nn.Module):
    """
    Per-column independent projection: column i gets its own W_i ∈ R^d and b_i ∈ R^d.

    e_i = W_i * x_i + b_i   (scalar → R^d, independent per column)

    Output: (B, n_tab) → (B, n_tab, d_model)
    """

    def __init__(self, n_tab: int, d_model: int):
        super().__init__()
        self.n_tab = n_tab
        self.d_model = d_model
        self.W = nn.Parameter(torch.empty(n_tab, d_model))
        self.b = nn.Parameter(torch.zeros(n_tab, d_model))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)


class TransformerTokenEncoder(nn.Module):
    """Stack of Transformer encoder layers over a token sequence (batch_first)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class FeatureWiseGating(nn.Module):
    """Sigmoid gate over concatenated modality vectors (same dim as input)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.lin(x))


class BiCrossAttentionBlock(nn.Module):
    """One layer of tab<->image bidirectional cross-attention with residuals."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        self.tab_q_img = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.img_q_tab = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.nt = nn.LayerNorm(d_model)
        self.ni = nn.LayerNorm(d_model)

    def forward(
        self, tab_tok: torch.Tensor, img_tok: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_out, w_tab = self.tab_q_img(tab_tok, img_tok, img_tok, average_attn_weights=False)
        tab2 = self.nt(tab_tok + t_out)
        i_out, w_img = self.img_q_tab(img_tok, tab2, tab2, average_attn_weights=False)
        img2 = self.ni(img_tok + i_out)
        return tab2, img2, w_tab, w_img


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention: ``queries`` attend over ``context`` (keys/values from context)."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, queries: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, attn = self.mha(queries, context, context, average_attn_weights=False)
        out = self.norm(queries + out)
        return out, attn
