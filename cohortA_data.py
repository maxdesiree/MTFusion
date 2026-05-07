"""
Cohort A (N=353): diagnostics workbook + denoised 12-lead CSV waveforms (FileName id).

Rhythm is encoded as an 11-dimensional multi-label vector (one-hot over classes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_RHYTHM_CLASSES: tuple[str, ...] = (
    "AF",
    "AFIB",
    "AT",
    "AVNRT",
    "AVRT",
    "SAAWR",
    "SA",
    "SB",
    "SR",
    "ST",
    "SVT",
)

RHYTHM_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(_RHYTHM_CLASSES)}


def rhythm_one_hot(series: pd.Series) -> np.ndarray:
    """(N, 11) float32 one-hot rows from Rhythm string column."""
    y = np.zeros((len(series), len(_RHYTHM_CLASSES)), dtype=np.float32)
    for i, v in enumerate(series.astype(str).str.strip()):
        if v not in RHYTHM_TO_IDX:
            raise ValueError(f"Unknown Rhythm label {v!r} at row {i}.")
        y[i, RHYTHM_TO_IDX[v]] = 1.0
    return y


def rhythm_one_hot_for_classes(series: pd.Series, classes: list[str]) -> np.ndarray:
    """(N, C) float32 one-hot for an explicit class list (in that order)."""
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(series), len(classes)), dtype=np.float32)
    for i, v in enumerate(series.astype(str).str.strip()):
        if v not in idx:
            raise ValueError(f"Unknown/filtered Rhythm label {v!r} at row {i}.")
        y[i, idx[v]] = 1.0
    return y


def tabular_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in ("FileName", "Rhythm", "Beat")]
    return cols


def encode_gender_series(s: pd.Series) -> pd.Series:
    """0/1 encoding; supports raw ``MALE``/``FEMALE`` strings or already-numeric CSVs (0/1)."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    u = s.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[u == "MALE"] = 0.0
    out[u == "FEMALE"] = 1.0
    return out


class CohortADenoisedDataset(Dataset):
    """Loads (T, 12) from cohortA_ECGDataDenoised/{FileName}.csv + tabular vector + multi-label y."""

    def __init__(
        self,
        df: pd.DataFrame,
        ecg_dir: Path,
        tab_cols: list[str],
        y: np.ndarray | None = None,
        target_len: int | None = 5000,
        lead_mean: np.ndarray | None = None,
        lead_std: np.ndarray | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.ecg_dir = Path(ecg_dir)
        self.tab_cols = tab_cols
        self.target_len = target_len
        self.lead_mean = lead_mean
        self.lead_std = lead_std
        self.y = rhythm_one_hot(self.df["Rhythm"]) if y is None else np.asarray(y, dtype=np.float32)
        self.tab = self.df[self.tab_cols].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def _load_ecg(self, stem: str) -> np.ndarray:
        p = self.ecg_dir / f"{stem}.csv"
        x = np.loadtxt(p, delimiter=",", dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 12:
            raise ValueError(f"Expected (T, 12) in {p}, got {x.shape}.")
        return x

    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        if self.target_len is None:
            return x
        t, c = x.shape
        if t == self.target_len:
            return x
        if t > self.target_len:
            return x[: self.target_len]
        pad = np.zeros((self.target_len - t, c), dtype=np.float32)
        return np.concatenate([x, pad], axis=0)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.lead_mean is None or self.lead_std is None:
            return x
        return (x - self.lead_mean) / self.lead_std

    def __getitem__(self, idx: int):
        stem = str(self.df.at[idx, "FileName"]).strip()
        x = self._load_ecg(stem)
        x = self._pad_or_crop(x)
        x = self._norm(x)
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.tab[idx]),
            torch.from_numpy(self.y[idx]),
        )


@torch.no_grad()
def compute_lead_stats_from_df(
    df: pd.DataFrame, ecg_dir: Path, target_len: int | None
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    for stem in df["FileName"].astype(str).str.strip():
        p = Path(ecg_dir) / f"{stem}.csv"
        x = np.loadtxt(p, delimiter=",", dtype=np.float32)
        if target_len is not None:
            if x.shape[0] >= target_len:
                x = x[:target_len]
            else:
                pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
        xs.append(x)
    X = np.concatenate(xs, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = X.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std
