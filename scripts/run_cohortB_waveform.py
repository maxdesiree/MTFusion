"""
run_cohortB_waveform.py
======================================
**Cohort B (digital ECG):** WFDB 12-lead waveforms + tabular fusion — separate from cohort A
(image / NPZ) and from **image** cohort B (``run_cohortB_stagebi_5splits_externalA.py``).

Reads waveforms from WFDB ``.hea`` + ``.dat`` under ``all_records/``, keyed by
``ValidateSet-DKD.xlsx`` (column ``Unnamed: 1``), plus tabular fields on the same sheet.
Default label column is ``stage_bi`` (override with ``--label-col`` if your sheet differs).

Methods:
- Concat (single-token ECG)
- Single-Attn (single-token ECG)
- MTFusion (multi-token ECG patches + tabular tokens; mean-pool per modality; fusion MLP; no cross-attention)
- MTFusion+cross-attn (bidirectional tab$\\leftrightarrow$ECG cross-attention + learned pooling; image-model analogue)

Evaluation (pick one):
- **K-fold CV** (default): ``--folds K`` with ``StratifiedKFold``.
- **Repeated stratified holdout** (e.g. 80/20): ``--holdout-repeats R --test-size 0.2`` with
  ``StratifiedShuffleSplit``; train on ``1 - test_size``, evaluate on the held-out fraction each repeat.
- **Bootstrap + OOB** (not a fixed 80/20): ``--bootstrap-iters B`` — stratified bootstrap training multiset
  and out-of-bag evaluation per replicate; mean ± SD across successful replicates.

Outputs (defaults: ``--results-dir`` / ``--file-prefix``):
- K-fold: ``{results-dir}/{prefix}_{label}.csv`` and ``.tex``
- Holdout: ``{results-dir}/{prefix}_{label}_holdout{train_pct}_{R}_summary.csv`` and ``.tex``
- Bootstrap: ``{results-dir}/{prefix}_{label}_bootstrap_summary.csv`` (optional per-iter CSV)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.waveform_fusion_models import (
    MTFusionWaveform,
    MTFusionWaveformCrossAttn,
    SingleTokenWaveformAttn,
    SingleTokenWaveformConcat,
)


class _TeeIO:
    """Duplicate writes to multiple text streams (e.g. console + log file)."""

    __slots__ = ("_streams",)

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


@dataclass(frozen=True)
class RowRef:
    record_rel: str  # e.g. "p1028/p10281780/s45863011/45863011"
    record_id: str   # e.g. "45863011"


def _parse_row_ref(row: pd.Series) -> RowRef:
    rel = str(row["Unnamed: 1"]).strip()
    base = Path(rel).name
    return RowRef(record_rel=rel, record_id=base)


def _load_record(all_records: Path, ref: RowRef) -> np.ndarray:
    """
    Returns waveform as float32 array (T, 12).
    """
    rec_path = all_records / ref.record_rel
    rec = wfdb.rdrecord(str(rec_path))
    x = rec.p_signal.astype(np.float32)  # (T, 12)
    # Safety: WFDB can sometimes yield non-finite samples depending on record.
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x


class WaveformTabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        all_records: Path,
        tab_cols: list[str],
        label_col: str,
        target_len: int | None = None,
        lead_mean: np.ndarray | None = None,
        lead_std: np.ndarray | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.all_records = all_records
        self.tab_cols = tab_cols
        self.label_col = label_col
        self.target_len = target_len
        self.lead_mean = lead_mean
        self.lead_std = lead_std

        self.refs = [_parse_row_ref(r) for _, r in self.df.iterrows()]
        self.y = self.df[label_col].astype(int).to_numpy()
        self.tab = self.df[tab_cols].astype(float).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.lead_mean is None or self.lead_std is None:
            return x
        return (x - self.lead_mean) / self.lead_std

    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        if self.target_len is None:
            return x
        T, C = x.shape
        if T == self.target_len:
            return x
        if T > self.target_len:
            return x[: self.target_len]
        pad = np.zeros((self.target_len - T, C), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx: int):
        x = _load_record(self.all_records, self.refs[idx])
        x = self._pad_or_crop(x)
        x = self._norm(x)
        return torch.from_numpy(x), torch.from_numpy(self.tab[idx]), torch.tensor(self.y[idx], dtype=torch.long)


@torch.no_grad()
def _compute_train_stats(df: pd.DataFrame, all_records: Path, target_len: int | None) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    for _, row in df.iterrows():
        ref = _parse_row_ref(row)
        x = _load_record(all_records, ref)
        if target_len is not None:
            if x.shape[0] >= target_len:
                x = x[:target_len]
            else:
                pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=0)
        xs.append(x)
    X = np.concatenate(xs, axis=0)  # (sum_T, 12)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = X.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def _train_one_epoch(model: nn.Module, dl: DataLoader, opt, device: torch.device) -> float:
    model.train()
    total = 0.0
    crit = nn.CrossEntropyLoss()
    for x_ecg, x_tab, y in dl:
        x_ecg, x_tab, y = x_ecg.to(device), x_tab.to(device), y.to(device)
        logits, _ = model(x_ecg, x_tab)
        loss = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.detach().item() * x_ecg.size(0)
    return total / len(dl.dataset)


@torch.no_grad()
def _eval(model: nn.Module, dl: DataLoader, device: torch.device, n_classes: int) -> dict[str, float]:
    model.eval()
    logits_all, y_all = [], []
    for x_ecg, x_tab, y in dl:
        logits, _ = model(x_ecg.to(device), x_tab.to(device))
        logits_all.append(logits.cpu())
        y_all.append(y.cpu())
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()

    try:
        if n_classes == 2:
            auroc = roc_auc_score(y, probs[:, 1])
        else:
            auroc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    f1_macro = f1_score(y, preds, average="macro", zero_division=0)
    out: dict[str, float] = {"auroc": float(auroc), "macro_f1": float(f1_macro)}

    if n_classes == 2:
        f1_per = f1_score(y, preds, average=None, zero_division=0)
        out["f1_class_0"] = float(f1_per[0])
        out["f1_class_1"] = float(f1_per[1])
        out["f1"] = float(f1_score(y, preds, pos_label=1, zero_division=0))
        out["accuracy"] = float(accuracy_score(y, preds))
        out["precision"] = float(precision_score(y, preds, pos_label=1, zero_division=0))
        out["recall"] = float(recall_score(y, preds, pos_label=1, zero_division=0))
        out["sensitivity"] = out["recall"]
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        out["TN"], out["FP"], out["FN"], out["TP"] = float(tn), float(fp), float(fn), float(tp)
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        min_lab = 0 if n0 <= n1 else 1
        try:
            if min_lab == 1:
                out["pr_auc_minority"] = float(average_precision_score(y, probs[:, 1]))
            else:
                out["pr_auc_minority"] = float(
                    average_precision_score((y == 0).astype(np.int64), probs[:, 0])
                )
        except ValueError:
            out["pr_auc_minority"] = float("nan")
        try:
            out["brier"] = float(brier_score_loss(y, probs[:, 1]))
        except ValueError:
            out["brier"] = float("nan")
    else:
        out["f1"] = float(f1_macro)
        out["accuracy"] = float(accuracy_score(y, preds))
        out["precision"] = out["recall"] = out["sensitivity"] = out["specificity"] = float("nan")
        out["f1_class_0"] = float("nan")
        out["f1_class_1"] = float("nan")
        out["pr_auc_minority"] = float("nan")
        out["brier"] = float("nan")
        out["TN"] = out["FP"] = out["FN"] = out["TP"] = float("nan")

    return out


def _metrics_row_for_csv(split: int, method: str, scope: str, met: dict[str, float]) -> dict:
    """Per-fold row: classification metrics (binary), plus confusion counts and AUROC / PR-AUC / Brier."""
    row: dict = {
        "split": split,
        "method": method,
        "scope": scope,
        "accuracy": met.get("accuracy", float("nan")),
        "precision": met.get("precision", float("nan")),
        "recall": met.get("recall", float("nan")),
        "sensitivity": met.get("sensitivity", float("nan")),
        "specificity": met.get("specificity", float("nan")),
        "f1": met.get("f1", float("nan")),
        "macro_f1": met["macro_f1"],
        "auroc": met["auroc"],
        "pr_auc_minority": met.get("pr_auc_minority", float("nan")),
        "brier": met.get("brier", float("nan")),
        "f1_class_0": met.get("f1_class_0", float("nan")),
        "f1_class_1": met.get("f1_class_1", float("nan")),
    }
    for k in ("TN", "FP", "FN", "TP"):
        v = met.get(k, float("nan"))
        row[k] = int(v) if np.isfinite(v) else ""
    return row


def stratified_bootstrap_train_oob(
    y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified bootstrap train indices (length N, with replacement) and OOB index array."""
    y = np.asarray(y).astype(int).ravel()
    classes = np.unique(y)
    if len(classes) < 2:
        raise SystemExit("Bootstrap requires at least two classes.")
    parts: list[np.ndarray] = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        parts.append(rng.choice(idx_c, size=len(idx_c), replace=True))
    tr = np.concatenate(parts)
    rng.shuffle(tr)
    cnt = np.bincount(tr, minlength=len(y))
    oob = np.where(cnt == 0)[0]
    return tr, oob


def _prepare_fold_dataloaders(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    *,
    tab_cols: list[str],
    label_col: str,
    all_records: Path,
    target_len: int | None,
    batch: int,
) -> tuple[DataLoader, DataLoader]:
    """Tab median imputation + scaler (train-only), per-lead waveform norm (train-only), DataLoaders."""
    df_tr = df_tr.copy()
    df_va = df_va.copy()

    tr_tab_raw = df_tr[tab_cols].astype(float)
    med = tr_tab_raw.median(axis=0, skipna=True)
    df_tr.loc[:, tab_cols] = tr_tab_raw.fillna(med).to_numpy(dtype=np.float32)
    df_va.loc[:, tab_cols] = df_va[tab_cols].astype(float).fillna(med).to_numpy(dtype=np.float32)

    sc_tab = StandardScaler().fit(df_tr[tab_cols].to_numpy(dtype=np.float32))
    Xtr_tab = sc_tab.transform(df_tr[tab_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    Xva_tab = sc_tab.transform(df_va[tab_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    for j, c in enumerate(tab_cols):
        df_tr[c] = Xtr_tab[:, j]
        df_va[c] = Xva_tab[:, j]

    mean, std = _compute_train_stats(df_tr, all_records, target_len)
    ds_tr = WaveformTabularDataset(
        df_tr, all_records, tab_cols, label_col, target_len=target_len, lead_mean=mean, lead_std=std
    )
    ds_va = WaveformTabularDataset(
        df_va, all_records, tab_cols, label_col, target_len=target_len, lead_mean=mean, lead_std=std
    )
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False)
    return dl_tr, dl_va


def _train_one_method_waveform(
    name: str,
    factory,
    *,
    n_tab: int,
    n_classes: int,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    epoch_log_interval: int = 10,
) -> dict[str, float]:
    model = factory(n_tab=n_tab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        loss = _train_one_epoch(model, dl_tr, opt, device)
        sched.step()
        if epoch_log_interval > 0 and ((ep + 1) % epoch_log_interval == 0 or ep == 0):
            met = _eval(model, dl_va, device, n_classes)
            print(
                f"  [{name:22s}] epoch {ep+1:3d} | loss={loss:.4f} | "
                f"AUROC={met['auroc']:.4f} | F1={met['macro_f1']:.4f}"
            )
    return _eval(model, dl_va, device, n_classes)


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """LaTeX table: wide summary (binary metrics) if present, else AUROC + macro-F1 only."""

    def row_wide(r: pd.Series, present: list[tuple[str, str]], bests: dict[str, float]) -> str:
        parts = [str(r["method"])]
        for base, _lab in present:
            mkey, skey = f"{base}_mean", f"{base}_std"
            m, s = float(r[mkey]), float(r[skey])
            cell = f"{m:.4f} $\\pm$ {s:.4f}"
            b = bests.get(base, float("nan"))
            if np.isfinite(b) and np.isfinite(m) and np.isclose(m, b):
                cell = f"\\textbf{{{cell}}}"
            parts.append(cell)
        return " & ".join(parts) + " \\\\"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    if len(df) > 0 and "accuracy_mean" in df.columns:
        specs = [
            ("accuracy", "Acc"),
            ("precision", "Prec"),
            ("recall", "Rec"),
            ("specificity", "Spec"),
            ("f1", "F1"),
            ("auroc", "AUC"),
            ("pr_auc_minority", "PR-AUC"),
            ("brier", "Brier"),
        ]
        present = [(b, lab) for b, lab in specs if f"{b}_mean" in df.columns]
        colspec = "l" + "c" * len(present)
        bests = {base: float(df[f"{base}_mean"].max()) for base, _ in present}
        lines.append("\\scriptsize")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append("\\toprule")
        lines.append("\\textbf{Method} & " + " & ".join(f"\\textbf{{{lab}}}" for _, lab in present) + " \\\\")
        lines.append("\\midrule")
        for _, r in df.iterrows():
            lines.append(row_wide(r, present, bests))
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("}% end resizebox")
    else:
        best_auroc = float(df["auroc_mean"].max()) if len(df) else float("nan")
        best_f1 = float(df["f1_mean"].max()) if len(df) else float("nan")

        def fmt(val: float, best: float) -> str:
            s = f"{val:.4f}"
            return f"\\textbf{{{s}}}" if np.isfinite(best) and np.isclose(val, best) else s

        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Method} & \\textbf{AUROC} & \\textbf{Macro-F1}\\\\")
        lines.append("\\midrule")
        for _, r in df.iterrows():
            au = fmt(float(r["auroc_mean"]), best_auroc)
            f1 = fmt(float(r["f1_mean"]), best_f1)
            lines.append(
                f"{r['method']} & {au} $\\pm$ {r['auroc_std']:.4f} & {f1} $\\pm$ {r['f1_std']:.4f}\\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=Path("ValidateSet-DKD.xlsx"))
    ap.add_argument("--sheet", type=str, default="Parameters")
    ap.add_argument("--all-records", type=Path, default=Path("all_records"))
    ap.add_argument(
        "--label-col",
        type=str,
        default="stage_bi",
        help="Binary/multiclass label column in the workbook (ValidateSet-DKD may use stage_bi).",
    )
    ap.add_argument("--tab-cols", type=str, default="age,gender,pulse")
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="StratifiedKFold splits (5 → ~80%% train / ~20%% validation per fold).",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--patch-len",
        type=int,
        default=250,
        help="Samples per ECG patch token for MTFusion (e.g. 250=0.5s at 500Hz); fewer tokens than 125 improves stability",
    )
    ap.add_argument("--stride", type=int, default=None, help="Stride between tokens (default=patch-len)")
    ap.add_argument("--target-len", type=int, default=5000, help="Pad/crop waveform to this length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--methods",
        type=str,
        default="Concat,Single-Attn,MTFusion,MTFusion+cross-attn",
        help="Comma-separated subset of: Concat, Single-Attn, MTFusion, MTFusion+cross-attn",
    )
    ap.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="If >0, use stratified bootstrap + OOB (no k-fold). Ignores --folds.",
    )
    ap.add_argument(
        "--min-oob",
        type=int,
        default=5,
        help="Skip a bootstrap draw if the OOB set is smaller than this.",
    )
    ap.add_argument(
        "--save-bootstrap-per-iter",
        type=Path,
        default=None,
        help="Optional CSV path for one row per method per replicate (bootstrap OOB or holdout repeats).",
    )
    ap.add_argument(
        "--holdout-repeats",
        type=int,
        default=0,
        help="If >0, StratifiedShuffleSplit this many times (--test-size held out); ignores --folds. Incompatible with --bootstrap-iters.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation fraction for --holdout-repeats (default 0.2 → 80%% train / 20%% eval).",
    )
    ap.add_argument(
        "--epoch-log-interval",
        type=int,
        default=10,
        help="Print val metrics every N epochs; 0 = only final metrics (useful for many bootstrap/holdout runs).",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cohortB_waveform"),
        help="Directory for cohort B (digital ECG) CSV/TeX outputs.",
    )
    ap.add_argument(
        "--file-prefix",
        type=str,
        default="cohortB_waveform",
        help="Stem for output files (e.g. cohortB_waveform_moderate.csv).",
    )
    ap.add_argument(
        "--eval-scope",
        type=str,
        default="internal_test_B",
        help="Value for the ``scope`` column in per-fold CSV (match image cohort for side-by-side tables).",
    )
    ap.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Per-split metrics CSV (default: {results-dir}/{prefix}_{label}_kfoldcv_per_fold.csv in k-fold mode).",
    )
    a = ap.parse_args()

    if a.bootstrap_iters > 0 and a.holdout_repeats > 0:
        raise SystemExit("Use only one of --bootstrap-iters or --holdout-repeats.")
    if not (0.0 < a.test_size < 1.0):
        raise SystemExit("--test-size must be in (0, 1).")
    if a.epoch_log_interval < 0:
        raise SystemExit("--epoch-log-interval must be >= 0.")

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    tab_cols = [c.strip() for c in a.tab_cols.split(",") if c.strip()]
    df = pd.read_excel(a.xlsx, sheet_name=a.sheet)
    df = df[df["Unnamed: 1"].notna() & df[a.label_col].notna()].copy()

    # Make labels zero-based contiguous
    raw = df[a.label_col].astype(int).to_numpy()
    uniq = np.unique(raw)
    mapping = {int(v): i for i, v in enumerate(sorted(uniq.tolist()))}
    df[a.label_col] = [mapping[int(v)] for v in raw]

    y = df[a.label_col].astype(int).to_numpy()
    n_classes = len(np.unique(y))

    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    folds = min(a.folds, max(2, min_count))
    if a.bootstrap_iters <= 0 and a.holdout_repeats <= 0 and folds != a.folds:
        print(f"Requested --folds={a.folds}, but min class count is {min_count}. Using folds={folds}.")
    print(f"Label column: {a.label_col}  classes: {sorted(np.unique(y).tolist())}  counts: {counts.tolist()}")
    print(f"Workbook: {a.xlsx.resolve()}  sheet={a.sheet!r}  n_rows={len(df)}")
    if a.bootstrap_iters <= 0 and a.holdout_repeats <= 0:
        print(
            f"CV: StratifiedKFold n_splits={folds} shuffle=True random_state={a.seed}  "
            f"(~{100.0 * (folds - 1) / folds:.1f}% train / ~{100.0 / folds:.1f}% validation per fold)"
        )

    _all_factories = {
        "Concat": lambda n_tab: SingleTokenWaveformConcat(n_tab=n_tab, n_classes=n_classes, d_model=a.d, dropout=a.dropout),
        "Single-Attn": lambda n_tab: SingleTokenWaveformAttn(
            n_tab=n_tab, n_classes=n_classes, d_model=a.d, n_heads=a.heads, dropout=a.dropout
        ),
        "MTFusion": lambda n_tab: MTFusionWaveform(
            n_tab=n_tab,
            n_classes=n_classes,
            d_model=None,
            n_heads=a.heads,
            dropout=a.dropout,
            patch_len=a.patch_len,
            stride=a.stride,
        ),
        "MTFusion+cross-attn": lambda n_tab: MTFusionWaveformCrossAttn(
            n_tab=n_tab,
            n_classes=n_classes,
            d_model=None,
            n_heads=a.heads,
            dropout=a.dropout,
            patch_len=a.patch_len,
            stride=a.stride,
        ),
    }
    method_names = [m.strip() for m in a.methods.split(",") if m.strip()]
    unknown = [m for m in method_names if m not in _all_factories]
    if unknown:
        raise SystemExit(f"--methods unknown: {unknown}. Expected subset of {list(_all_factories)}.")
    methods = {name: _all_factories[name] for name in method_names}

    device = torch.device(a.device)
    per_method = {m: {"auroc": [], "macro_f1": []} for m in methods}
    per_iter_rows: list[dict] = []
    eli = a.epoch_log_interval

    if a.bootstrap_iters > 0:
        print(f"\nBootstrap mode: {a.bootstrap_iters} replicates, min-OOB={a.min_oob} (no k-fold CV).\n")
        brng = np.random.default_rng(a.seed)
        skipped = 0
        ok_iters = 0
        for b in range(a.bootstrap_iters):
            tr_boot, oob = stratified_bootstrap_train_oob(y, brng)
            if len(oob) < a.min_oob or len(np.unique(y[oob])) < 2:
                skipped += 1
                continue
            print(
                f"\n{'='*52}\nBootstrap {b + 1}/{a.bootstrap_iters}  "
                f"train_multiset={len(tr_boot)}  |OOB|={len(oob)}\n{'='*52}"
            )
            df_tr = df.iloc[tr_boot].reset_index(drop=True)
            df_oob = df.iloc[oob].reset_index(drop=True)
            dl_tr, dl_oob = _prepare_fold_dataloaders(
                df_tr,
                df_oob,
                tab_cols=tab_cols,
                label_col=a.label_col,
                all_records=a.all_records,
                target_len=a.target_len,
                batch=a.batch,
            )
            torch.manual_seed(a.seed + b * 100_003)
            for name, factory in methods.items():
                met = _train_one_method_waveform(
                    name,
                    factory,
                    n_tab=len(tab_cols),
                    n_classes=n_classes,
                    dl_tr=dl_tr,
                    dl_va=dl_oob,
                    device=device,
                    epochs=a.epochs,
                    lr=a.lr,
                    epoch_log_interval=eli,
                )
                per_method[name]["auroc"].append(met["auroc"])
                per_method[name]["macro_f1"].append(met["macro_f1"])
                per_iter_rows.append(_metrics_row_for_csv(b + 1, name, a.eval_scope, met))
                print(
                    f"  [{name:22s}] OOB  AUROC={met['auroc']:.4f}  macro_F1={met['macro_f1']:.4f}",
                    flush=True,
                )
            ok_iters += 1
        print(f"\nSuccessful bootstrap replicates: {ok_iters}  skipped: {skipped}", flush=True)
        if ok_iters == 0:
            raise SystemExit("No successful bootstrap draws; lower --min-oob or increase data size.")
    elif a.holdout_repeats > 0:
        train_pct = int(round((1.0 - float(a.test_size)) * 100))
        print(
            f"\nStratified holdout: {a.holdout_repeats} repeats, "
            f"~{train_pct}% train / {int(round(float(a.test_size) * 100))}% eval (test_size={a.test_size}).\n",
            flush=True,
        )
        sss = StratifiedShuffleSplit(
            n_splits=a.holdout_repeats,
            test_size=a.test_size,
            random_state=a.seed,
        )
        for rep_i, (tr_idx, va_idx) in enumerate(sss.split(np.zeros((len(y), 1)), y), start=1):
            if rep_i == 1 or rep_i % 25 == 0 or rep_i == a.holdout_repeats:
                print(f"--- Holdout repeat {rep_i}/{a.holdout_repeats} ---", flush=True)
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[va_idx].copy()
            dl_tr, dl_va = _prepare_fold_dataloaders(
                df_tr,
                df_va,
                tab_cols=tab_cols,
                label_col=a.label_col,
                all_records=a.all_records,
                target_len=a.target_len,
                batch=a.batch,
            )
            torch.manual_seed(a.seed + rep_i * 100_003)
            for name, factory in methods.items():
                final = _train_one_method_waveform(
                    name,
                    factory,
                    n_tab=len(tab_cols),
                    n_classes=n_classes,
                    dl_tr=dl_tr,
                    dl_va=dl_va,
                    device=device,
                    epochs=a.epochs,
                    lr=a.lr,
                    epoch_log_interval=eli,
                )
                per_method[name]["auroc"].append(final["auroc"])
                per_method[name]["macro_f1"].append(final["macro_f1"])
                per_iter_rows.append(_metrics_row_for_csv(rep_i, name, a.eval_scope, final))
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=a.seed)
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(df, y), start=1):
            tr_n, va_n = len(tr_idx), len(va_idx)
            print(
                f"\n{'='*52}\nFold {fold_i}/{folds}  |train|={tr_n}  |val|={va_n}  "
                f"(~{100.0 * tr_n / (tr_n + va_n):.1f}% / ~{100.0 * va_n / (tr_n + va_n):.1f}%)\n{'='*52}"
            )
            df_tr = df.iloc[tr_idx].copy()
            df_va = df.iloc[va_idx].copy()
            dl_tr, dl_va = _prepare_fold_dataloaders(
                df_tr,
                df_va,
                tab_cols=tab_cols,
                label_col=a.label_col,
                all_records=a.all_records,
                target_len=a.target_len,
                batch=a.batch,
            )
            for name, factory in methods.items():
                final = _train_one_method_waveform(
                    name,
                    factory,
                    n_tab=len(tab_cols),
                    n_classes=n_classes,
                    dl_tr=dl_tr,
                    dl_va=dl_va,
                    device=device,
                    epochs=a.epochs,
                    lr=a.lr,
                    epoch_log_interval=eli,
                )
                per_method[name]["auroc"].append(final["auroc"])
                per_method[name]["macro_f1"].append(final["macro_f1"])
                per_iter_rows.append(_metrics_row_for_csv(fold_i, name, a.eval_scope, final))

    # Build summary table (mean ± std over folds / replicates)
    ddf = pd.DataFrame(per_iter_rows) if per_iter_rows else pd.DataFrame()
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "macro_f1",
        "auroc",
        "pr_auc_minority",
        "brier",
        "f1_class_0",
        "f1_class_1",
    ]
    rows = []
    for name in methods:
        row: dict = {"method": name}
        if len(ddf) > 0 and "method" in ddf.columns:
            sub = ddf[ddf["method"] == name]
            for col in metric_cols:
                if col not in sub.columns:
                    continue
                v = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                row[f"{col}_mean"] = float(np.nanmean(v))
                row[f"{col}_std"] = (
                    float(np.nanstd(v, ddof=1)) if np.sum(np.isfinite(v)) > 1 else 0.0
                )
        else:
            au = np.array(per_method[name]["auroc"], dtype=float)
            f1v = np.array(per_method[name]["macro_f1"], dtype=float)
            row["auroc_mean"] = float(np.nanmean(au))
            row["auroc_std"] = float(np.nanstd(au, ddof=1)) if len(au) > 1 else 0.0
            row["macro_f1_mean"] = float(np.nanmean(f1v))
            row["macro_f1_std"] = float(np.nanstd(f1v, ddof=1)) if len(f1v) > 1 else 0.0
        rows.append(row)
    out_df = pd.DataFrame(rows)
    if len(ddf) == 0 and "macro_f1_mean" in out_df.columns:
        out_df = out_df.rename(columns={"macro_f1_mean": "f1_mean", "macro_f1_std": "f1_std"})

    results_dir = a.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_label = str(a.label_col).replace(" ", "_")
    prefix = str(a.file_prefix).strip() or "cohortB_waveform"

    if per_iter_rows:
        if a.detail_csv is not None:
            detail_path = Path(a.detail_csv)
        elif a.bootstrap_iters > 0:
            detail_path = results_dir / f"{prefix}_{safe_label}_bootstrap_per_replicate.csv"
        elif a.holdout_repeats > 0:
            hp = int(round((1.0 - float(a.test_size)) * 100))
            detail_path = results_dir / f"{prefix}_{safe_label}_holdout{hp}_n{a.holdout_repeats}_per_replicate.csv"
        else:
            detail_path = results_dir / f"{prefix}_{safe_label}_kfoldcv_per_fold.csv"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_iter_rows).to_csv(detail_path, index=False)
        print(f"Wrote per-split metrics CSV: {detail_path}")
        if a.save_bootstrap_per_iter is not None:
            alt = Path(a.save_bootstrap_per_iter)
            alt.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(per_iter_rows).to_csv(alt, index=False)
            print(f"Wrote copy: {alt}")

    if a.bootstrap_iters > 0:
        csv_path = results_dir / f"{prefix}_{safe_label}_bootstrap_summary.csv"
        out_df.to_csv(csv_path, index=False)
        summ_note = "bootstrap OOB replicates"
    elif a.holdout_repeats > 0:
        train_pct = int(round((1.0 - float(a.test_size)) * 100))
        eval_pct = int(round(float(a.test_size) * 100))
        stem = f"{prefix}_{safe_label}_holdout{train_pct}_n{a.holdout_repeats}"
        csv_path = results_dir / f"{stem}_summary.csv"
        tex_path = results_dir / f"{stem}.tex"
        out_df.to_csv(csv_path, index=False)
        tex_path.write_text(
            _latex_table(
                out_df,
                caption=(
                    f"Cohort B (digital ECG): waveform + tabular fusion (label: {a.label_col}). "
                    f"Mean $\\pm$ std over {a.holdout_repeats} stratified "
                    f"{train_pct}/{eval_pct} train/val splits."
                ),
                label=f"tab:{stem}",
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {tex_path}")
        summ_note = f"{a.holdout_repeats} stratified holdout repeats (~{train_pct}% train)"
    else:
        csv_path = results_dir / f"{prefix}_{safe_label}.csv"
        tex_path = results_dir / f"{prefix}_{safe_label}.tex"
        out_df.to_csv(csv_path, index=False)
        tex_path.write_text(
            _latex_table(
                out_df,
                caption=(
                    f"Cohort B (digital ECG): WFDB waveforms + tabular fusion (label: {a.label_col}). "
                    f"Five-fold stratified CV; mean $\\pm$ std. "
                    f"Recall equals sensitivity for the positive class (DKD). "
                    f"F1 is the class-1 F1-score; macro-F1 is in the summary CSV."
                ),
                label=f"tab:{prefix}_{safe_label}",
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {tex_path}")
        summ_note = "folds"

    if per_iter_rows:
        print("\n" + "=" * 60)
        print("PER-SPLIT METRICS (accuracy, precision, recall, sensitivity, specificity, F1, AUROC, PR-AUC, Brier, …)")
        print("=" * 60)
        print(pd.DataFrame(per_iter_rows).to_string(index=False))
        print("=" * 60)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS (mean ± std over {summ_note})")
    print("=" * 60)
    print(out_df.to_string(index=False))
    print("=" * 60)
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()

