#!/usr/bin/env python3
"""
**Cohort — RGB ECG scans (images), not WFDB waveforms.** For digital 12-lead waveform + tabular
fusion, use ``scripts/run_cohortB_waveform.py`` (or ``run_waveform_manuscript_experiments.py``).

Cohort (ImageDKD, stage_bi): stratified cross-validation on the internal cohort.

- Default: **StratifiedKFold** with ``n_splits=5`` (each fold ~80%% train / 20%% validation).
- Deep models: optional **early stopping** on the validation fold (default: maximize val AUROC,
  patience 10 epochs, ``--early-stopping-patience 0`` to disable); best weights are restored.
- Use ``--torch-methods MTFusion,MTFusion+cross-attn`` with ``--merge-csv prior.csv`` to retrain only
  a subset of deep models and copy the other deep rows from the prior CSV.
- **MTFusion**: baseline-matched head (same MLP as Concat); spatial mean-pool over patch tokens +
  tab TokenEmbedder mean-pool. ``--mtfusion-img-layers`` / ``--mtfusion-tab-layers`` apply to
  **MTFusion+cross-attn** only. Backbone frozen by default; ``--train-mtfusion-backbone`` trains
  ResNet50 for MTFusion / MTFusion+cross-attn. All deep models share ``--lr-image`` /
  ``--lr-tabular`` / ``--lr-fusion``.
- Optional: ``--cv-mode shuffle`` for five independent StratifiedShuffleSplit draws and
  ``--test-size`` (e.g. 0.2).
- ``--splitter-random-state`` sets the sklearn StratifiedKFold / ShuffleSplit ``random_state`` only
  (default: 42, independent of ``--seed``). Change ``--seed`` alone to rerun with new PyTorch /
  training RNG while keeping the same fold boundaries (needed when using ``--merge-csv``).

Classical baselines: 129-D (scaled 120-D geometry + tabular), unless
``--deep-only``.

Deep models: ResNet50 on RGB ECG scans (224×224) + tabular — Concat, Single-Attn, MTFusion,
MTFusion+cross-attn.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.prepare_dataset_npz import _infer_ecg_image_path  # noqa: E402
from scripts.mtfusion_resnet_models import (  # noqa: E402
    MTFusionResNet,
    MTFusionResNetCrossAttention,
    SingleTokenAttnResNet,
    SingleTokenConcatResNet,
)

TORCH_METHODS: tuple[str, ...] = ("Concat", "Single-Attn", "MTFusion", "MTFusion+cross-attn")
# Deterministic per-method offset (avoid PYTHONHASHSEED-dependent hash()).
_TORCH_SEED_OFF: dict[str, int] = {
    "Concat": 11,
    "Single-Attn": 17,
    "MTFusion": 23,
    "MTFusion+cross-attn": 29,
}


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


def _parse_only_folds_arg(s: str) -> frozenset[int] | None:
    """1-based fold indices to run (e.g. '1' or '1,3'). Empty = all folds."""
    if not s or not str(s).strip():
        return None
    out: list[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return frozenset(out)


def _parse_torch_methods_arg(s: str) -> frozenset[str]:
    if not s or not str(s).strip():
        return frozenset(TORCH_METHODS)
    names = [p.strip() for p in str(s).split(",") if p.strip()]
    unknown = set(names) - set(TORCH_METHODS)
    if unknown:
        raise SystemExit(f"--torch-methods unknown name(s): {sorted(unknown)}. Expected subset of {list(TORCH_METHODS)}.")
    return frozenset(names)


def _row_from_merge_csv(prev: pd.DataFrame, split_i: int, method: str) -> dict[str, object]:
    m = (prev["split"] == split_i) & (prev["method"] == method)
    sub = prev.loc[m]
    if len(sub) != 1:
        raise SystemExit(
            f"--merge-csv: need exactly one row for split={split_i} method={method!r}, found {len(sub)}."
        )
    return {k: sub.iloc[0][k] for k in sub.columns}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_paths(npz_path: Path, xlsx: Path, sheet: str, image_dir: Path) -> list[Path]:
    """Same row order as prepare_dataset_npz.py with --ecg-source=image (skips rows with no image)."""
    db = np.load(npz_path, allow_pickle=True)
    n = len(db["labels"])
    label_col = str(db["label_col"][0])
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df[df["ecg_filename"].notna()].copy()
    df = df[df[label_col].notna()].copy()
    paths: list[Path] = []
    for _, row in df.iterrows():
        ip = _infer_ecg_image_path(row, image_dir)
        if ip is None:
            continue
        paths.append(ip)
    if len(paths) != n:
        raise RuntimeError(
            f"Resolved {len(paths)} image paths but npz has N={n}. "
            "Rebuild data/cohortB_stage_bi.npz with the same workbook and image directory, or fix paths."
        )
    return paths


def _inverse_freq_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int).ravel()
    classes = np.unique(y)
    w = np.zeros(len(y), dtype=np.float64)
    n = len(y)
    for c in classes:
        m = y == c
        w[m] = n / (len(classes) * max(1, int(m.sum())))
    return w


def compute_metrics(y: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int).ravel()
    probs = np.asarray(probs)
    preds = np.asarray(preds).astype(int).ravel()
    out: dict[str, float] = {}
    try:
        out["auroc"] = float(roc_auc_score(y, probs[:, 1]))
    except ValueError:
        out["auroc"] = float("nan")
    out["macro_f1"] = float(f1_score(y, preds, average="macro", zero_division=0))
    f1s = f1_score(y, preds, average=None, zero_division=0)
    out["f1_class_0"] = float(f1s[0]) if len(f1s) > 0 else 0.0
    out["f1_class_1"] = float(f1s[1]) if len(f1s) > 1 else 0.0
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    y_minor = (y == minority).astype(int)
    out["pr_auc_minority"] = float(average_precision_score(y_minor, probs[:, minority]))
    out["brier"] = float(brier_score_loss(y, probs[:, 1]))
    return out


def confusion_binary(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, preds, labels=[0, 1]).astype(np.int64)


class ECGImageTabDataset(Dataset):
    def __init__(self, paths: list[Path], X_tab: np.ndarray, y: np.ndarray, image_size: int = 224):
        self.paths = paths
        self.X_tab = np.asarray(X_tab, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.image_size = image_size
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = Image.open(self.paths[i]).convert("RGB")
        im = im.resize((self.image_size, self.image_size), Image.BICUBIC)
        t = T.functional.to_tensor(im)
        t = self.normalize(t)
        return t, torch.from_numpy(self.X_tab[i]), torch.tensor(self.y[i], dtype=torch.long)


def train_one_epoch_img(model: nn.Module, loader: DataLoader, opt, device: torch.device) -> float:
    model.train()
    total = 0.0
    crit = nn.CrossEntropyLoss()
    for img, x_tab, y in loader:
        img, x_tab, y = img.to(device), x_tab.to(device), y.to(device)
        logits, _ = model(img, x_tab)
        loss = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.detach().item() * img.size(0)
    return total / len(loader.dataset)


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_to_device(
    model: nn.Module, state: dict[str, torch.Tensor], device: torch.device
) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


@torch.no_grad()
def eval_torch_img(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], np.ndarray, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    logits_all, y_all = [], []
    loss_sum, n_ex = 0.0, 0
    for img, x_tab, y in loader:
        img_d, tab_d, y_d = img.to(device), x_tab.to(device), y.to(device)
        logits, _ = model(img_d, tab_d)
        loss_sum += float(crit(logits, y_d).item()) * img.size(0)
        n_ex += int(img.size(0))
        logits_all.append(logits.cpu())
        y_all.append(y.cpu())
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    met = compute_metrics(y, probs, preds)
    cm = confusion_binary(y, preds)
    val_loss = loss_sum / max(1, n_ex)
    return met, cm, val_loss


def _early_stop_score(met: dict[str, float], val_loss: float, metric: str) -> float:
    if metric == "loss":
        return val_loss
    if metric == "macro_f1":
        return met["macro_f1"]
    return met["auroc"]


def _is_better(score: float, best: float | None, metric: str, min_delta: float) -> bool:
    if metric == "loss":
        if math.isnan(score):
            return False
        if best is None:
            return True
        return score < best - min_delta
    if math.isnan(score):
        return False
    if best is None:
        return True
    return score > best + min_delta


def run_logreg(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, *, seed: int
) -> tuple[dict[str, float], np.ndarray]:
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=8000,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(X_tr.astype(np.float64), y_tr.astype(int))
    probs = clf.predict_proba(X_te.astype(np.float64))
    preds = clf.predict(X_te.astype(np.float64))
    return compute_metrics(y_te, probs, preds), confusion_binary(y_te, preds)


def run_xgb(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, *, seed: int
) -> tuple[dict[str, float], np.ndarray]:
    if xgb is None:
        raise RuntimeError("xgboost not installed")
    sw = _inverse_freq_sample_weight(y_tr)
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=seed,
        tree_method="hist",
        n_jobs=0,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    clf.fit(X_tr.astype(np.float64), y_tr.astype(int), sample_weight=sw)
    probs = clf.predict_proba(X_te.astype(np.float64))
    preds = clf.predict(X_te.astype(np.float64))
    return compute_metrics(y_te, probs, preds), confusion_binary(y_te, preds)


@dataclass(frozen=True)
class Cfg:
    cohort_b_npz: Path
    xlsx: Path
    sheet: str
    image_dir: Path
    splits: int
    test_size: float
    cv_mode: str
    deep_only: bool
    epochs: int
    batch: int
    seed: int
    splitter_random_state: int
    device: str
    lr_image: float
    lr_tabular: float
    lr_fusion: float
    lr: float
    d: int
    heads: int
    dropout: float
    results_dir: Path
    run_tag: str
    export_tex: bool
    pretrained_resnet: bool
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_min_delta: float
    torch_methods: frozenset[str]
    merge_csv: Path | None
    mtfusion_img_layers: int
    mtfusion_tab_layers: int
    train_mtfusion_backbone: bool
    only_folds: frozenset[int] | None


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-b", type=Path, default=Path("data/cohortB_stage_bi.npz"))
    ap.add_argument("--xlsx", type=Path, default=Path("ValidateSet-ImageDKD.xlsx"))
    ap.add_argument("--sheet", type=str, default="Parameters")
    ap.add_argument("--image-dir", type=Path, default=Path("ProcessedData"))
    ap.add_argument("--splits", type=int, default=5, help="Number of CV folds (StratifiedKFold) or shuffle splits.")
    ap.add_argument(
        "--cv-mode",
        type=str,
        default="kfold",
        choices=("kfold", "shuffle"),
        help="kfold: StratifiedKFold; shuffle: StratifiedShuffleSplit with --test-size.",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction when --cv-mode shuffle.")
    ap.add_argument("--deep-only", action="store_true", help="Train only PyTorch / ResNet models (skip LR/XGB).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--splitter-random-state",
        type=int,
        default=42,
        help="random_state for StratifiedKFold / StratifiedShuffleSplit only (default: 42).",
    )
    ap.add_argument("--device", type=str, default="cpu")
    # Default deep LRs (image / tabular / fusion heads): 0.0001 / 0.0005 / 0.0003
    ap.add_argument("--lr-image", type=float, default=0.0001)
    ap.add_argument("--lr-tabular", type=float, default=0.0005)
    ap.add_argument("--lr-fusion", type=float, default=0.0003)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-pretrained-resnet", action="store_true")
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop deep training if validation metric does not improve for this many epochs (0=off).",
    )
    ap.add_argument(
        "--early-stopping-metric",
        type=str,
        default="auroc",
        choices=("auroc", "macro_f1", "loss"),
        help="Metric on the validation fold: maximize AUROC/macro-F1 or minimize mean val cross-entropy.",
    )
    ap.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum change to count as improvement (AUROC/F1: additive; loss: subtractive).",
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--run-tag", type=str, default="cohortB_stagebi_5foldcv_resnet")
    ap.add_argument("--no-export-tex", action="store_true")
    ap.add_argument(
        "--torch-methods",
        type=str,
        default="",
        help=f"Comma-separated deep models to train (default: all). Choices: {', '.join(TORCH_METHODS)}.",
    )
    ap.add_argument(
        "--merge-csv",
        type=Path,
        default=None,
        help="When --torch-methods is a strict subset, copy non-trained deep rows from this CSV (same splits).",
    )
    ap.add_argument(
        "--mtfusion-img-layers",
        type=int,
        default=1,
        help="MTFusion+cross-attn: Transformer layers over image patch tokens (default 1).",
    )
    ap.add_argument(
        "--mtfusion-tab-layers",
        type=int,
        default=1,
        help="MTFusion+cross-attn: Transformer layers over tabular tokens (default 1).",
    )
    ap.add_argument(
        "--train-mtfusion-backbone",
        action="store_true",
        help="If set, train ResNet50 backbone for MTFusion / MTFusion+cross-attn (default: frozen).",
    )
    ap.add_argument(
        "--only-folds",
        type=str,
        default="",
        help="Comma-separated 1-based fold indices to train/eval (e.g. '1'). Other folds are copied "
        "from --merge-csv (required). Uses the same split as full n_splits CV.",
    )
    a = ap.parse_args()

    if a.mtfusion_img_layers < 1 or a.mtfusion_tab_layers < 1:
        raise SystemExit("--mtfusion-img-layers and --mtfusion-tab-layers must be >= 1.")

    torch_sel = _parse_torch_methods_arg(a.torch_methods)
    only_folds = _parse_only_folds_arg(a.only_folds)
    if torch_sel != frozenset(TORCH_METHODS):
        if a.merge_csv is None or not a.merge_csv.is_file():
            raise SystemExit(
                "When using a strict --torch-methods subset, pass --merge-csv pointing to a CSV "
                "that already contains rows for the other deep models (same fold indices)."
            )
    if only_folds is not None:
        if a.merge_csv is None or not a.merge_csv.is_file():
            raise SystemExit("--only-folds requires --merge-csv (used to fill non-selected folds).")

    cfg = Cfg(
        cohort_b_npz=a.cohort_b,
        xlsx=a.xlsx,
        sheet=a.sheet,
        image_dir=a.image_dir,
        splits=a.splits,
        test_size=a.test_size,
        cv_mode=a.cv_mode,
        deep_only=a.deep_only,
        epochs=a.epochs,
        batch=a.batch,
        seed=a.seed,
        splitter_random_state=a.splitter_random_state,
        device=a.device,
        lr_image=a.lr_image,
        lr_tabular=a.lr_tabular,
        lr_fusion=a.lr_fusion,
        lr=a.lr,
        d=a.d,
        heads=a.heads,
        dropout=a.dropout,
        results_dir=a.results_dir,
        run_tag=a.run_tag,
        export_tex=not a.no_export_tex,
        pretrained_resnet=not a.no_pretrained_resnet,
        early_stopping_patience=a.early_stopping_patience,
        early_stopping_metric=a.early_stopping_metric,
        early_stopping_min_delta=a.early_stopping_min_delta,
        torch_methods=torch_sel,
        merge_csv=a.merge_csv if a.merge_csv and a.merge_csv.is_file() else None,
        mtfusion_img_layers=a.mtfusion_img_layers,
        mtfusion_tab_layers=a.mtfusion_tab_layers,
        train_mtfusion_backbone=a.train_mtfusion_backbone,
        only_folds=only_folds,
    )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    db = np.load(cfg.cohort_b_npz, allow_pickle=True)
    Xe = db["ecg"].astype(np.float32)
    Xt = db["vitals"].astype(np.float32)
    y = db["labels"].astype(int)
    n_tab = Xt.shape[1]
    n_classes = int(len(np.unique(y)))
    if n_classes != 2:
        raise SystemExit("Expected binary labels.")

    paths = build_image_paths(cfg.cohort_b_npz, cfg.xlsx, cfg.sheet, cfg.image_dir)

    prev_deep: pd.DataFrame | None = None
    if cfg.merge_csv is not None:
        prev_deep = pd.read_csv(cfg.merge_csv)
        print(f"Loaded --merge-csv {cfg.merge_csv} for non-trained deep rows.", flush=True)

    if cfg.torch_methods != frozenset(TORCH_METHODS):
        print(
            "Training deep subset: "
            + ", ".join(m for m in TORCH_METHODS if m in cfg.torch_methods),
            flush=True,
        )

    rows: list[dict[str, object]] = []
    if cfg.cv_mode == "kfold":
        splitter = StratifiedKFold(
            n_splits=cfg.splits, shuffle=True, random_state=cfg.splitter_random_state
        )
        split_iter = list(splitter.split(Xe, y))
        print(
            f"CV mode: StratifiedKFold n_splits={cfg.splits} (~{(1 - 1 / cfg.splits) * 100:.0f}% train per fold), "
            f"splitter_random_state={cfg.splitter_random_state}.",
            flush=True,
        )
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=cfg.splits,
            test_size=cfg.test_size,
            random_state=cfg.splitter_random_state,
        )
        split_iter = list(splitter.split(Xe, y))
        print(
            f"CV mode: StratifiedShuffleSplit n_splits={cfg.splits} test_size={cfg.test_size} "
            f"splitter_random_state={cfg.splitter_random_state}.",
            flush=True,
        )

    if cfg.only_folds is not None:
        bad = sorted(k for k in cfg.only_folds if k < 1 or k > cfg.splits)
        if bad:
            raise SystemExit(f"--only-folds indices must be in 1..{cfg.splits}; invalid: {bad}")
        print(f"Only running folds: {sorted(cfg.only_folds)} (others from merge-csv).", flush=True)

    factories: dict[str, type] = {
        "Concat": SingleTokenConcatResNet,
        "Single-Attn": SingleTokenAttnResNet,
        "MTFusion": MTFusionResNet,
        "MTFusion+cross-attn": MTFusionResNetCrossAttention,
    }

    for split_i, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        if cfg.only_folds is not None and split_i not in cfg.only_folds:
            assert prev_deep is not None
            sub = prev_deep.loc[prev_deep["split"] == split_i]
            if len(sub) == 0:
                raise SystemExit(f"--merge-csv: no rows with split={split_i} to copy for skipped fold.")
            for _, r in sub.iterrows():
                rows.append({k: r[k] for k in sub.columns})
            print(
                f"\n{'='*60}\nFold {split_i}/{cfg.splits}  SKIPPED — copied {len(sub)} rows from merge-csv\n{'='*60}",
                flush=True,
            )
            continue

        y_tr_f, y_te_f = y[tr_idx], y[te_idx]
        print(
            f"\n{'='*60}\nFold {split_i}/{cfg.splits}  train={len(tr_idx)}  val={len(te_idx)}  "
            f"pos_train={int(y_tr_f.sum())}  pos_val={int(y_te_f.sum())}\n{'='*60}",
            flush=True,
        )

        sc_e = StandardScaler().fit(Xe[tr_idx])
        sc_t = StandardScaler().fit(Xt[tr_idx])
        Xe_tr = sc_e.transform(Xe[tr_idx]).astype(np.float32)
        Xt_tr = sc_t.transform(Xt[tr_idx]).astype(np.float32)
        y_tr = y[tr_idx]
        Xe_te = sc_e.transform(Xe[te_idx]).astype(np.float32)
        Xt_te = sc_t.transform(Xt[te_idx]).astype(np.float32)
        y_te = y[te_idx]

        Xflat_tr = np.hstack([Xe_tr, Xt_tr]).astype(np.float32)
        Xflat_te = np.hstack([Xe_te, Xt_te]).astype(np.float32)
        seed_f = cfg.seed + split_i * 17

        if not cfg.deep_only:
            met, cm = run_logreg(Xflat_tr, y_tr, Xflat_te, y_te, seed=seed_f)
            rows.append(
                {
                    "split": split_i,
                    "method": "LogReg-L2",
                    "scope": "internal_test_B",
                    **met,
                    "TN": int(cm[0, 0]),
                    "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]),
                    "TP": int(cm[1, 1]),
                }
            )
            print(
                f"  [LogReg-L2] AUROC={met['auroc']:.4f}  macro_F1={met['macro_f1']:.4f}  "
                f"TN,FP,FN,TP={cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}",
                flush=True,
            )

            if xgb is not None:
                met, cm = run_xgb(Xflat_tr, y_tr, Xflat_te, y_te, seed=seed_f)
                rows.append(
                    {
                        "split": split_i,
                        "method": "XGBoost",
                        "scope": "internal_test_B",
                        **met,
                        "TN": int(cm[0, 0]),
                        "FP": int(cm[0, 1]),
                        "FN": int(cm[1, 0]),
                        "TP": int(cm[1, 1]),
                    }
                )
                print(
                    f"  [XGBoost   ] AUROC={met['auroc']:.4f}  macro_F1={met['macro_f1']:.4f}  "
                    f"TN,FP,FN,TP={cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}",
                    flush=True,
                )

        tr_paths = [paths[i] for i in tr_idx]
        te_paths = [paths[i] for i in te_idx]
        ds_tr = ECGImageTabDataset(tr_paths, Xt_tr, y_tr)
        ds_te = ECGImageTabDataset(te_paths, Xt_te, y_te)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, num_workers=0, pin_memory=False)
        dl_te = DataLoader(ds_te, batch_size=cfg.batch, shuffle=False, num_workers=0, pin_memory=False)

        # Shared width d for Concat / Single-Attn / MTFusion (MTFusion+cross-attn uses d_model=d).
        d_model = max(128, cfg.d)

        for method in TORCH_METHODS:
            if method not in cfg.torch_methods:
                assert prev_deep is not None
                copied = _row_from_merge_csv(prev_deep, split_i, method)
                rows.append(copied)
                print(
                    f"  [{method:18s}] skip train — copied from merge-csv  "
                    f"AUROC={float(copied['auroc']):.4f}  macro_F1={float(copied['macro_f1']):.4f}",
                    flush=True,
                )
                continue

            fold_seed = cfg.seed + split_i * 1_000 + _TORCH_SEED_OFF[method]
            set_seed(fold_seed)
            cls = factories[method]
            if method == "Concat":
                model = cls(n_tab=n_tab, n_classes=n_classes, d=d_model, dropout=cfg.dropout, pretrained=cfg.pretrained_resnet)
            elif method == "Single-Attn":
                model = cls(
                    n_tab=n_tab,
                    n_classes=n_classes,
                    d=d_model,
                    n_heads=cfg.heads,
                    dropout=cfg.dropout,
                    pretrained=cfg.pretrained_resnet,
                )
            elif method == "MTFusion":
                model = cls(
                    n_tab=n_tab,
                    n_classes=n_classes,
                    d=d_model,
                    dropout=cfg.dropout,
                    pretrained=cfg.pretrained_resnet,
                    freeze_backbone=not cfg.train_mtfusion_backbone,
                )
            else:
                # method == "MTFusion+cross-attn"
                d_mtf = d_model
                h_mtf = _valid_heads(d_mtf, cfg.heads)
                model = cls(
                    n_tab=n_tab,
                    n_classes=n_classes,
                    d_model=d_mtf,
                    n_heads=h_mtf,
                    img_layers=cfg.mtfusion_img_layers,
                    tab_layers=cfg.mtfusion_tab_layers,
                    dropout=cfg.dropout,
                    pretrained=cfg.pretrained_resnet,
                    freeze_backbone=not cfg.train_mtfusion_backbone,
                )

            model = model.to(device)
            opt = torch.optim.AdamW(
                model.get_param_groups(cfg.lr_image, cfg.lr_tabular, cfg.lr_fusion),
                weight_decay=1e-4,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
            use_es = cfg.early_stopping_patience > 0
            best_state: dict[str, torch.Tensor] | None = None
            best_score: float | None = None
            best_ep = 0
            bad_epochs = 0
            stopped_early = False

            for ep in range(cfg.epochs):
                loss_tr = train_one_epoch_img(model, dl_tr, opt, device)
                sched.step()
                met_log, _, val_loss = eval_torch_img(model, dl_te, device)
                score = _early_stop_score(met_log, val_loss, cfg.early_stopping_metric)
                if use_es:
                    if _is_better(score, best_score, cfg.early_stopping_metric, cfg.early_stopping_min_delta):
                        best_score = score
                        best_state = _state_dict_cpu(model)
                        best_ep = ep + 1
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= cfg.early_stopping_patience:
                            stopped_early = True
                            print(
                                f"  [{method:18s}] early stop at ep {ep + 1} "
                                f"(no improvement in {cfg.early_stopping_patience} epochs; "
                                f"best {cfg.early_stopping_metric} @ ep {best_ep})",
                                flush=True,
                            )
                            break

                log_ep = use_es or (ep + 1) % 10 == 0 or ep == 0
                if log_ep:
                    print(
                        f"  [{method:18s}] ep {ep+1:3d} loss_tr={loss_tr:.4f} val_loss={val_loss:.4f} "
                        f"AUROC={met_log['auroc']:.4f} F1m={met_log['macro_f1']:.4f}",
                        flush=True,
                    )

            if use_es and best_state is not None:
                _load_state_dict_to_device(model, best_state, device)
            met, cm, _ = eval_torch_img(model, dl_te, device)
            rows.append(
                {
                    "split": split_i,
                    "method": method,
                    "scope": "internal_test_B",
                    **met,
                    "TN": int(cm[0, 0]),
                    "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]),
                    "TP": int(cm[1, 1]),
                }
            )
            es_note = f" best@{best_ep}" if use_es and best_ep > 0 else ""
            if stopped_early:
                es_note += " (early-stop)"
            print(
                f"  [{method:18s}] FINAL{es_note}  AUROC={met['auroc']:.4f}  macro_F1={met['macro_f1']:.4f}  "
                f"F1_0={met['f1_class_0']:.4f}  F1_1={met['f1_class_1']:.4f}  "
                f"TN,FP,FN,TP={cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}",
                flush=True,
            )

    out_path = cfg.results_dir / f"{cfg.run_tag}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nWrote {out_path}", flush=True)

    if cfg.export_tex:
        tex_script = _ROOT / "scripts" / "export_cohortB_internal_metrics_tex.py"
        if tex_script.is_file():
            import subprocess

            cmd = [sys.executable, str(tex_script), "--csv", str(out_path)]
            if cfg.cohort_b_npz.is_file():
                cmd.extend(["--cohort-b-npz", str(cfg.cohort_b_npz)])
            subprocess.run(cmd, check=True)
        else:
            print(f"Note: skip LaTeX export (missing {tex_script}).", flush=True)


if __name__ == "__main__":
    main()
