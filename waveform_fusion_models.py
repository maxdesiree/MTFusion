"""
waveform_fusion_models.py
=========================
Waveform-based versions of the manuscript baselines and MTFusion (primary multi-token model).

Key design choice (per user request):
- Single-token ECG baseline: ECG encoder outputs ONE token (global embedding).
- Multi-token ECG proposed: ECG encoder outputs MANY tokens (patch tokens).

Tabular modality:
- Treated as a multi-token sequence (one token per scalar column) using
  `TokenEmbedder` from `scripts/mtfusion.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.mtfusion import (
    BiCrossAttentionBlock,
    CrossModalAttention,
    FeatureWiseGating,
    TokenEmbedder,
    TransformerTokenEncoder,
)


def _valid_heads(d_model: int, requested: int) -> int:
    """Pick a MultiheadAttention head count that divides d_model."""
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}.")
    req = max(1, int(requested))
    req = min(req, d_model)
    for h in range(req, 0, -1):
        if d_model % h == 0:
            return h
    return 1


class ECGGlobalEncoder(nn.Module):
    """Simple 1D CNN → global vector (single-token ECG)."""

    def __init__(self, n_leads: int = 12, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) waveform where C=n_leads
        returns: (B, d_model)
        """
        x = x.transpose(1, 2)  # (B, C, T)
        h = self.net(x)        # (B, d, T')
        z = h.mean(dim=-1)     # global average pool
        return self.norm(z)


class ECGTokenEncoder(nn.Module):
    """1D Conv patchifier → token sequence (multi-token ECG)."""

    def __init__(
        self,
        n_leads: int = 12,
        d_model: int = 64,
        patch_len: int = 125,
        stride: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if stride is None:
            stride = patch_len
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(
            in_channels=n_leads,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, n_tokens, d_model)
        """
        x = x.transpose(1, 2)           # (B, C, T)
        t = self.proj(x)                # (B, d, n_tokens)
        t = t.transpose(1, 2)           # (B, n_tokens, d)
        t = self.norm(t)
        return self.dropout(t)


class SingleTokenWaveformConcat(nn.Module):
    """Baseline: ECG global vector + tabular global vector → concat → classifier."""

    def __init__(self, n_tab: int, n_classes: int, d_model: int = 64, dropout: float = 0.1, n_leads: int = 12):
        super().__init__()
        self.ecg = ECGGlobalEncoder(n_leads=n_leads, d_model=d_model, dropout=dropout)
        self.tab = nn.Sequential(nn.Linear(n_tab, d_model), nn.GELU(), nn.Dropout(dropout))
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor):
        z_ecg = self.ecg(x_ecg)        # (B, d)
        z_tab = self.tab(x_tab)        # (B, d)
        z = self.fuse(torch.cat([z_ecg, z_tab], dim=-1))
        return self.cls(z), None


class SingleTokenWaveformAttn(nn.Module):
    """Baseline: self-attn over 2 tokens (ECG global token + tab global token)."""

    def __init__(
        self,
        n_tab: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_leads: int = 12,
    ):
        super().__init__()
        self.ecg = ECGGlobalEncoder(n_leads=n_leads, d_model=d_model, dropout=dropout)
        self.tab = nn.Sequential(nn.Linear(n_tab, d_model), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor):
        t_ecg = self.ecg(x_ecg).unsqueeze(1)  # (B,1,d)
        t_tab = self.tab(x_tab).unsqueeze(1)  # (B,1,d)
        seq = torch.cat([t_ecg, t_tab], dim=1)
        out, _ = self.attn(seq, seq, seq)
        out = self.norm(out + seq)
        z = out.mean(dim=1)
        return self.cls(z), None


class MTFusionWaveform(nn.Module):
    """
    Primary MTFusion (waveform): multi-token ECG patches + multi-token tabular columns,
    modality-wise mean pooling, then the same style 2-layer fusion MLP as image MTFusionResNet
    (no cross-attention between modalities before fusion).
    """

    def __init__(
        self,
        n_tab: int,
        n_classes: int,
        d_model: int | None = None,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_leads: int = 12,
        patch_len: int = 250,
        stride: int | None = None,
    ):
        super().__init__()
        if d_model is None:
            d_model = int(n_tab)
        d_model = int(d_model)
        self.ecg_tokens = ECGTokenEncoder(
            n_leads=n_leads, d_model=d_model, patch_len=patch_len, stride=stride, dropout=dropout
        )
        self.norm_ecg = nn.LayerNorm(d_model)
        self.tab_tokens = TokenEmbedder(n_tab, d_model)
        self.norm_tab = nn.LayerNorm(d_model)
        d_fuse = max(1, d_model // 2)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(d_fuse, n_classes)

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor):
        e = self.norm_ecg(self.ecg_tokens(x_ecg))
        z_ecg = e.mean(dim=1)
        v = self.norm_tab(self.tab_tokens(x_tab))
        z_tab = v.mean(dim=1)
        z = self.head(torch.cat([z_ecg, z_tab], dim=-1))
        return self.cls(z), None


class MTFusionWaveformCrossAttn(nn.Module):
    """
    MTFusion + cross-attention (waveform analogue of image MTFusionResNetCrossAttention).

    ECG patch tokens and tabular tokens each pass a shallow Transformer encoder, then two
    bidirectional cross-attention blocks (tabular $\\leftrightarrow$ ECG), learned softmax
    pooling per modality, gated fusion MLP, and classifier.
    """

    def __init__(
        self,
        n_tab: int,
        n_classes: int,
        d_model: int | None = None,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_leads: int = 12,
        patch_len: int = 250,
        stride: int | None = None,
        xattn_layers: int = 2,
        attn_pool_tau: float = 0.5,
    ):
        super().__init__()
        if d_model is None:
            d_model = int(n_tab)
        d_model = int(d_model)
        n_heads = _valid_heads(d_model, int(n_heads))
        self.ecg_tokens = ECGTokenEncoder(
            n_leads=n_leads, d_model=d_model, patch_len=patch_len, stride=stride, dropout=dropout
        )
        self.ecg_enc = TransformerTokenEncoder(d_model, n_heads, n_layers=1, dropout=dropout)
        self.tab_tokens = TokenEmbedder(n_tab, d_model)
        self.tab_enc = TransformerTokenEncoder(d_model, n_heads, n_layers=1, dropout=dropout)
        self.norm_ecg = nn.LayerNorm(d_model)
        self.norm_tab = nn.LayerNorm(d_model)
        self.ecg_pool = nn.Linear(d_model, 1)
        self.tab_pool = nn.Linear(d_model, 1)
        self.register_buffer("attn_pool_tau", torch.tensor(float(attn_pool_tau)), persistent=False)
        self.xattn = nn.ModuleList(
            [BiCrossAttentionBlock(d_model, n_heads, dropout) for _ in range(int(xattn_layers))]
        )
        self.fuse_gate = FeatureWiseGating(2 * d_model)
        d_fuse = max(1, d_model // 2)
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(d_fuse, n_classes)

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor):
        ecg_tok = self.ecg_tokens(x_ecg)
        ecg_tok = self.ecg_enc(ecg_tok)
        ecg_tok = self.norm_ecg(ecg_tok)

        tab_tok = self.tab_enc(self.tab_tokens(x_tab))
        tab_tok = self.norm_tab(tab_tok)

        attn_last = None
        for blk in self.xattn:
            tab_tok, ecg_tok, w_ab, _ = blk(tab_tok, ecg_tok)
            attn_last = w_ab

        tau = self.attn_pool_tau.clamp(min=1e-6)
        le = self.ecg_pool(ecg_tok)
        le = le - le.max(dim=1, keepdim=True).values
        w_e = torch.softmax(le / tau, dim=1)
        z_ecg = (ecg_tok * w_e).sum(dim=1)

        lt = self.tab_pool(tab_tok)
        lt = lt - lt.max(dim=1, keepdim=True).values
        w_t = torch.softmax(lt / tau, dim=1)
        z_tab = (tab_tok * w_t).sum(dim=1)

        z = self.fuse_gate(torch.cat([z_ecg, z_tab], dim=-1))
        z = self.fuse(z)
        return self.cls(z), attn_last


class MTFusionWaveformUnidirectionalCross(nn.Module):
    """
    Lighter variant: tabular tokens query ECG tokens once (CrossModalAttention), gated tab pool,
    plus global ECG mean shortcut.  Kept for ablations; the runner uses ``MTFusionWaveformCrossAttn``.
    """

    def __init__(
        self,
        n_tab: int,
        n_classes: int,
        d_model: int | None = None,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_leads: int = 12,
        patch_len: int = 250,
        stride: int | None = None,
    ):
        super().__init__()
        if d_model is None:
            d_model = int(n_tab)
        d_model = int(d_model)
        n_heads = _valid_heads(d_model, int(n_heads))
        self.ecg_tokens = ECGTokenEncoder(
            n_leads=n_leads, d_model=d_model, patch_len=patch_len, stride=stride, dropout=dropout
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ecg_transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.tab_tokens = TokenEmbedder(n_tab, d_model)
        self.pre_norm_E = nn.LayerNorm(d_model)
        self.pre_norm_V = nn.LayerNorm(d_model)
        self.cross = CrossModalAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.tab_gate = nn.Linear(d_model, 1)
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x_ecg: torch.Tensor, x_tab: torch.Tensor):
        E = self.ecg_tokens(x_ecg)
        E = self.ecg_transformer(E)
        e_global = E.mean(dim=1)

        V = self.tab_tokens(x_tab)
        E = self.pre_norm_E(E)
        V = self.pre_norm_V(V)
        C, attn = self.cross(V, E)
        C = self.cross_dropout(C)

        w = torch.softmax(self.tab_gate(C), dim=1)
        z_local = (C * w).sum(dim=1)
        z = z_local + e_global
        return self.cls(z), attn

