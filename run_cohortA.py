"""
Cohort A (N=353) multi-label scaling experiments (Steps 2, 4, 5).

- Waveforms: denoised CSV (12 leads, length 5000) under ``cohortA_ECGDataDenoised``.
- Labels: one-hot Rhythm (11 classes), ``BCEWithLogitsLoss``; sigmoid only at evaluation.
- Models: Concat, Single-Attn, MTFusion, MTFusion+cross-attn from ``waveform_fusion_models``.
- Evaluation: repeated stratified 80/20 holdout (default 5 repeats), no leakage, preprocessing fit on train only.

Default training uses **40 epochs** with **early stopping** (patience 10), plus
repeated stratified **80/20 holdout** (5 repeats).

Example::

  .venv/bin/python scripts/prepare_cohortA_scaling_subsets.py
  .venv/bin/python scripts/run_cohortA_multilabel_scaling.py --device cpu
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cohortA_data import (
    CohortADenoisedDataset,
    compute_lead_stats_from_df,
    encode_gender_series,
    rhythm_one_hot_for_classes,
    tabular_columns,
)
from scripts.waveform_fusion_models import (
    MTFusionWaveform,
    MTFusionWaveformCrossAttn,
    SingleTokenWaveformAttn,
    SingleTokenWaveformConcat,
)


def _collect_param_ids(*modules: nn.Module | None) -> set[int]:
    ids: set[int] = set()
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            ids.add(id(p))
    return ids


def adamw_param_groups(model: nn.Module, lr_img: float, lr_tab: float, lr_fuse: float) -> list[dict]:
    """Assign LRs by modality branch (ECG vs tabular vs fusion), matching manuscript protocol."""
    ecg_mods: list[nn.Module | None] = []
    tab_mods: list[nn.Module | None] = []
    for name in ("ecg", "ecg_tokens", "ecg_enc", "ecg_transformer"):
        if hasattr(model, name):
            ecg_mods.append(getattr(model, name))
    for name in ("tab", "tab_tokens", "tab_enc"):
        if hasattr(model, name):
            tab_mods.append(getattr(model, name))

    ecg_ids = _collect_param_ids(*ecg_mods)
    tab_ids = _collect_param_ids(*tab_mods)
    g_img, g_tab, g_fuse = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in ecg_ids:
            g_img.append(p)
        elif pid in tab_ids:
            g_tab.append(p)
        else:
            g_fuse.append(p)
    groups = []
    if g_img:
        groups.append({"params": g_img, "lr": float(lr_img)})
    if g_tab:
        groups.append({"params": g_tab, "lr": float(lr_tab)})
    if g_fuse:
        groups.append({"params": g_fuse, "lr": float(lr_fuse)})
    return groups


@torch.no_grad()
def multilabel_metrics(logits: torch.Tensor, y_true: np.ndarray) -> dict[str, float]:
    """logits: (N, C) raw; y_true: (N, C) binary {0,1}."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    y = y_true.astype(np.float32)
    preds = (probs >= 0.5).astype(np.int32)

    out: dict[str, float] = {}
    out["macro_f1"] = float(f1_score(y, preds, average="macro", zero_division=0))
    out["micro_f1"] = float(f1_score(y, preds, average="micro", zero_division=0))
    out["macro_precision"] = float(precision_score(y, preds, average="macro", zero_division=0))
    out["macro_recall"] = float(recall_score(y, preds, average="macro", zero_division=0))

    try:
        out["macro_auroc"] = float(roc_auc_score(y, probs, average="macro"))
    except ValueError:
        out["macro_auroc"] = float("nan")

    pr_vals: list[float] = []
    for c in range(y.shape[1]):
        yt = y[:, c]
        if float(yt.sum()) <= 0.0 or float(yt.sum()) >= float(len(yt)):
            continue
        try:
            pr_vals.append(float(average_precision_score(yt, probs[:, c])))
        except ValueError:
            continue
    out["macro_pr_auc"] = float(np.mean(pr_vals)) if pr_vals else float("nan")

    # Exact match ratio (subset accuracy)
    out["accuracy_exact"] = float(accuracy_score(y, preds))

    # Hamming accuracy (label-wise)
    out["hamming_accuracy"] = float((y == preds).mean())

    return out


def _filter_rare_rhythm_classes(df: pd.DataFrame, *, min_count: int = 10) -> tuple[pd.DataFrame, list[str]]:
    vc = df["Rhythm"].astype(str).value_counts()
    keep = [c for c, n in vc.items() if int(n) >= int(min_count)]
    df2 = df[df["Rhythm"].astype(str).isin(set(keep))].copy()
    # stable deterministic order: most frequent first, then alpha for ties
    vc2 = df2["Rhythm"].astype(str).value_counts()
    keep_sorted = sorted(keep, key=lambda k: (-int(vc2.get(k, 0)), str(k)))
    return df2.reset_index(drop=True), keep_sorted


def _keep_labels_with_train_val_support(y_tr: np.ndarray, y_va: np.ndarray) -> list[int]:
    """Keep label columns that have at least one positive in BOTH train and val (needed for PR-AUC / stable ES)."""
    keep: list[int] = []
    for c in range(y_tr.shape[1]):
        if y_tr[:, c].sum() >= 1 and y_va[:, c].sum() >= 1:
            keep.append(c)
    return keep


def _prepare_fold_loaders(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    *,
    ecg_dir: Path,
    tab_cols: list[str],
    y_tr: np.ndarray,
    y_va: np.ndarray,
    target_len: int | None,
    batch: int,
) -> tuple[DataLoader, DataLoader]:
    df_tr = df_tr.copy()
    df_va = df_va.copy()

    tr_tab = df_tr[tab_cols].astype(float)
    med = tr_tab.median(axis=0, skipna=True)
    df_tr.loc[:, tab_cols] = tr_tab.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
        dtype=np.float32
    )
    df_va.loc[:, tab_cols] = (
        df_va[tab_cols].astype(float).fillna(med).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    )

    sc = StandardScaler().fit(df_tr[tab_cols].to_numpy(dtype=np.float32))
    Xtr = sc.transform(df_tr[tab_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    Xva = sc.transform(df_va[tab_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    for j, c in enumerate(tab_cols):
        df_tr[c] = Xtr[:, j]
        df_va[c] = Xva[:, j]

    mean, std = compute_lead_stats_from_df(df_tr, ecg_dir, target_len)
    ds_tr = CohortADenoisedDataset(
        df_tr, ecg_dir, tab_cols, y=y_tr, target_len=target_len, lead_mean=mean, lead_std=std
    )
    ds_va = CohortADenoisedDataset(
        df_va, ecg_dir, tab_cols, y=y_va, target_len=target_len, lead_mean=mean, lead_std=std
    )
    return (
        DataLoader(ds_tr, batch_size=batch, shuffle=True),
        DataLoader(ds_va, batch_size=batch, shuffle=False),
    )


def _train_one_epoch(model: nn.Module, dl: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    crit = nn.BCEWithLogitsLoss()
    tot = 0.0
    n = 0
    for x_ecg, x_tab, y in dl:
        x_ecg = x_ecg.to(device)
        x_tab = x_tab.to(device)
        y = y.to(device)
        logits, _ = model(x_ecg, x_tab)
        loss = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += float(loss.detach()) * x_ecg.size(0)
        n += x_ecg.size(0)
    return tot / max(1, n)


@torch.no_grad()
def _eval_logits(model: nn.Module, dl: DataLoader, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x_ecg, x_tab, y in dl:
        logits, _ = model(x_ecg.to(device), x_tab.to(device))
        logits_all.append(logits.cpu())
        y_all.append(y.numpy())
    return torch.cat(logits_all, dim=0), np.concatenate(y_all, axis=0)


def train_with_early_stopping(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    early_stopping_patience: int,
    lr_img: float,
    lr_tab: float,
    lr_fuse: float,
) -> dict[str, float]:
    opt = torch.optim.AdamW(adamw_param_groups(model, lr_img, lr_tab, lr_fuse), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_state = None
    best_score = -1.0
    bad = 0
    use_es = int(early_stopping_patience) > 0
    for ep in range(epochs):
        _train_one_epoch(model, dl_tr, opt, device)
        sched.step()
        logits, y = _eval_logits(model, dl_va, device)
        met = multilabel_metrics(logits, y)
        prv = float(met["macro_pr_auc"]) if np.isfinite(met["macro_pr_auc"]) else float("nan")
        f1v = float(met["macro_f1"]) if np.isfinite(met["macro_f1"]) else float("nan")
        if np.isfinite(prv) and np.isfinite(f1v):
            monitor = 0.5 * (prv + f1v)
        elif np.isfinite(prv):
            monitor = prv
        elif np.isfinite(f1v):
            monitor = f1v
        else:
            monitor = -1.0
        if monitor > best_score:
            best_score = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if use_es and bad >= int(early_stopping_patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    logits, y = _eval_logits(model, dl_va, device)
    return multilabel_metrics(logits, y)


def _parse_dataset_tag(path: Path) -> tuple[int, str | None]:
    stem = path.stem
    m = re.match(r"dataset_(\d+)$", stem)
    if m:
        return int(m.group(1)), None
    m2 = re.match(r"dataset_(\d+)_seed(\d+)$", stem)
    if m2:
        return int(m2.group(1)), m2.group(2)
    return -1, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "cohortA_scaling_v2")
    ap.add_argument("--ecg-dir", type=Path, default=_ROOT / "cohortA_ECGDataDenoised")
    ap.add_argument("--xlsx", type=Path, default=None, help="Optional: diagnostics xlsx (unused if CSV subsets exist)")
    ap.add_argument("--holdout-repeats", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-class-count", type=int, default=10, help="Drop Rhythm classes with < this many samples per subset.")
    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Max epochs per fold (cosine schedule T_max matches this).",
    )
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop if val monitor (0.5·macro-PR-AUC+macro-F1, else whichever is finite) does not improve for this many epochs; "
        "0 = disabled (run all --epochs).",
    )
    ap.add_argument("--lr-image", type=float, default=1e-4)
    ap.add_argument("--lr-tab", type=float, default=5e-4)
    ap.add_argument("--lr-fusion", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-len", type=int, default=250)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--target-len", type=int, default=5000)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--torch-seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--methods",
        type=str,
        default="Concat,Single-Attn,MTFusion,MTFusion+cross-attn",
    )
    ap.add_argument("--results-dir", type=Path, default=_ROOT / "results" / "cohortA_scaling_v2")
    ap.add_argument(
        "--dataset-files",
        type=str,
        default="",
        help="Comma-separated stems under --data-dir (e.g. dataset_100_seed1.csv). Empty = all dataset_*.csv.",
    )
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Missing --data-dir={data_dir}. Run scripts/prepare_cohortA_scaling_subsets.py first.")

    if str(a.dataset_files).strip():
        names = [x.strip() for x in str(a.dataset_files).split(",") if x.strip()]
        csvs = [data_dir / n for n in names]
        missing = [str(p) for p in csvs if not p.is_file()]
        if missing:
            raise SystemExit(f"Missing dataset files: {missing}")
    else:
        csvs = sorted(data_dir.glob("dataset_*.csv"))
        csvs = sorted(csvs, key=lambda p: (_parse_dataset_tag(p)[0], _parse_dataset_tag(p)[1] or ""))
    if not csvs:
        raise SystemExit(f"No dataset_*.csv under {data_dir}")

    device = torch.device(a.device)
    method_names = [m.strip() for m in a.methods.split(",") if m.strip()]

    results_dir = Path(a.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict] = []

    for csv_path in csvs:
        size_n, seed_tag = _parse_dataset_tag(csv_path)
        if size_n < 0:
            continue
        df = pd.read_csv(csv_path)
        df["Gender"] = encode_gender_series(df["Gender"])
        tab_cols = tabular_columns(df)
        for c in tab_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df_f, classes = _filter_rare_rhythm_classes(df, min_count=int(a.min_class_count))
        if len(classes) < 2:
            print(f"{csv_path.name}: skipped (after filtering <{a.min_class_count}, classes={len(classes)}).")
            continue
        y = rhythm_one_hot_for_classes(df_f["Rhythm"], classes)
        idx_all = np.arange(len(df_f), dtype=int)
        rh = df_f["Rhythm"].astype(str).to_numpy()

        for rep_i in range(1, int(a.holdout_repeats) + 1):
            # Mutually-exclusive ``Rhythm``: stratify on the string label (stable multi-class stratification).
            rs = int(a.split_seed) + rep_i * 10_009 + (abs(hash(csv_path.name)) % 100_003)
            tr_idx, va_idx = train_test_split(
                idx_all,
                test_size=float(a.test_size),
                stratify=rh,
                random_state=rs,
            )
            df_tr = df_f.iloc[tr_idx].copy()
            df_va = df_f.iloc[va_idx].copy()
            y_tr_full = y[tr_idx]
            y_va_full = y[va_idx]
            keep_cols = _keep_labels_with_train_val_support(y_tr_full, y_va_full)
            if len(keep_cols) < 2:
                print(
                    f"{csv_path.name} rep{rep_i}: skipped (need >=2 labels with train+val positives; got {len(keep_cols)})."
                )
                continue

            y_tr = y_tr_full[:, keep_cols]
            y_va = y_va_full[:, keep_cols]
            classes_rep = [classes[i] for i in keep_cols]
            n_classes = len(keep_cols)

            dl_tr, dl_va = _prepare_fold_loaders(
                df_tr,
                df_va,
                ecg_dir=Path(a.ecg_dir),
                tab_cols=tab_cols,
                y_tr=y_tr,
                y_va=y_va,
                target_len=a.target_len,
                batch=int(a.batch),
            )
            n_tab = len(tab_cols)

            factories: dict[str, callable] = {
                "Concat": lambda: SingleTokenWaveformConcat(
                    n_tab=n_tab, n_classes=n_classes, d_model=a.d, dropout=a.dropout
                ),
                "Single-Attn": lambda: SingleTokenWaveformAttn(
                    n_tab=n_tab, n_classes=n_classes, d_model=a.d, n_heads=a.heads, dropout=a.dropout
                ),
                "MTFusion": lambda: MTFusionWaveform(
                    n_tab=n_tab,
                    n_classes=n_classes,
                    d_model=None,
                    n_heads=a.heads,
                    dropout=a.dropout,
                    patch_len=a.patch_len,
                    stride=a.stride,
                ),
                "MTFusion+cross-attn": lambda: MTFusionWaveformCrossAttn(
                    n_tab=n_tab,
                    n_classes=n_classes,
                    d_model=None,
                    n_heads=a.heads,
                    dropout=a.dropout,
                    patch_len=a.patch_len,
                    stride=a.stride,
                ),
            }
            for name in method_names:
                if name not in factories:
                    raise SystemExit(f"Unknown method {name!r}")
                torch.manual_seed(int(a.torch_seed) + rep_i * 17 + hash(name) % 10_007)
                model = factories[name]().to(device)
                met = train_with_early_stopping(
                    model,
                    dl_tr,
                    dl_va,
                    device,
                    epochs=int(a.epochs),
                    early_stopping_patience=int(a.early_stopping_patience),
                    lr_img=float(a.lr_image),
                    lr_tab=float(a.lr_tab),
                    lr_fuse=float(a.lr_fusion),
                )
                row = {
                    "dataset_file": csv_path.name,
                    "dataset_size": size_n,
                    "subset_seed_tag": seed_tag or "",
                    "repeat": rep_i,
                    "method": name,
                    "n_classes_used": int(n_classes),
                    "classes_used": "|".join(classes_rep),
                    **{k: float(v) for k, v in met.items()},
                }
                detail_rows.append(row)
                print(
                    f"{csv_path.name} rep{rep_i} {name}: micro_f1={met['micro_f1']:.4f} "
                    f"macro_pr_auc={met['macro_pr_auc']:.4f}"
                )

    if not detail_rows:
        raise SystemExit("No successful runs (all subsets/repeats skipped?). Check logs above.")
    det = pd.DataFrame(detail_rows)
    det_path = results_dir / "cohortA_scaling_per_repeat.csv"
    det.to_csv(det_path, index=False)
    print(f"Wrote {det_path}")

    # Summary: mean ± std over repeats (and seeds when multiple files per size)
    summ_rows: list[dict] = []
    for size_n in sorted(det["dataset_size"].unique()):
        sub = det[det["dataset_size"] == size_n]
        for name in method_names:
            s2 = sub[sub["method"] == name]
            if len(s2) == 0:
                continue
            def ms(col: str) -> tuple[float, float]:
                v = pd.to_numeric(s2[col], errors="coerce").to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return float("nan"), float("nan")
                return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0

            mi_m, mi_s = ms("micro_f1")
            pr_m, pr_s = ms("macro_pr_auc")
            mf_m, mf_s = ms("macro_f1")
            summ_rows.append(
                {
                    "Dataset size": int(size_n),
                    "Model": name,
                    "Micro-F1": f"{mi_m:.4f} ± {mi_s:.4f}",
                    "Macro PR-AUC": f"{pr_m:.4f} ± {pr_s:.4f}",
                    "Macro-F1": f"{mf_m:.4f} ± {mf_s:.4f}",
                    "macro_f1_mean": mf_m,
                    "macro_f1_std": mf_s,
                    "micro_f1_mean": mi_m,
                    "micro_f1_std": mi_s,
                    "macro_pr_auc_mean": pr_m,
                    "macro_pr_auc_std": pr_s,
                    "n_runs": int(len(s2)),
                }
            )
    summ = pd.DataFrame(summ_rows)
    summ_path = results_dir / "cohortA_scaling_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"Wrote {summ_path}")
    print("\nDataset size\tModel\tMicro-F1\tMacro PR-AUC")
    for _, r in summ.sort_values(["Dataset size", "Model"]).iterrows():
        print(f"{int(r['Dataset size'])}\t{r['Model']}\t{r['Micro-F1']}\t{r['Macro PR-AUC']}")


if __name__ == "__main__":
    main()
