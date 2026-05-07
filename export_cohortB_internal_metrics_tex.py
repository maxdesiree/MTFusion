#!/usr/bin/env python3
"""
Build ``results/cohortB_internal_metrics_table.tex`` from a cohort-B evaluation CSV
plus classical baselines (LogReg, XGBoost) on the same StratifiedKFold splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None


def _inverse_freq_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int).ravel()
    classes = np.unique(y)
    w = np.zeros(len(y), dtype=np.float64)
    n = len(y)
    for c in classes:
        m = y == c
        w[m] = n / (len(classes) * max(1, int(m.sum())))
    return w


def _compute_metrics(y: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, float]:
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


def _confusion_binary(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, preds, labels=[0, 1]).astype(np.int64)


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
    return _compute_metrics(y_te, probs, preds), _confusion_binary(y_te, preds)


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
    return _compute_metrics(y_te, probs, preds), _confusion_binary(y_te, preds)


def _fmt(x: float) -> str:
    if x != x:  # nan
        return "---"
    return f"{x:.3f}"


def _fmt_mean_std(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        return "---"
    return f"{a.mean():.3f}$\\pm${a.std(ddof=1):.3f}"


def classical_rows(
    npz_path: Path, *, splits: int, seed: int
) -> list[dict[str, object]]:
    db = np.load(npz_path, allow_pickle=True)
    Xe = db["ecg"].astype(np.float32)
    Xt = db["vitals"].astype(np.float32)
    y = db["labels"].astype(int)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for split_i, (tr_idx, te_idx) in enumerate(splitter.split(Xe, y), start=1):
        sc_e = StandardScaler().fit(Xe[tr_idx])
        sc_t = StandardScaler().fit(Xt[tr_idx])
        Xe_tr = sc_e.transform(Xe[tr_idx]).astype(np.float32)
        Xt_tr = sc_t.transform(Xt[tr_idx]).astype(np.float32)
        Xe_te = sc_e.transform(Xe[te_idx]).astype(np.float32)
        Xt_te = sc_t.transform(Xt[te_idx]).astype(np.float32)
        y_tr, y_te = y[tr_idx], y[te_idx]
        Xflat_tr = np.hstack([Xe_tr, Xt_tr]).astype(np.float32)
        Xflat_te = np.hstack([Xe_te, Xt_te]).astype(np.float32)
        seed_f = seed + split_i * 17
        met_lr, cm_lr = run_logreg(Xflat_tr, y_tr, Xflat_te, y_te, seed=seed_f)
        TN, FP, FN, TP = int(cm_lr[0, 0]), int(cm_lr[0, 1]), int(cm_lr[1, 0]), int(cm_lr[1, 1])
        rows.append(
            {
                "split": split_i,
                "method": "LogReg-L2",
                "scope": "internal_test_B",
                **met_lr,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "TP": TP,
            }
        )
        try:
            met, cm = run_xgb(Xflat_tr, y_tr, Xflat_te, y_te, seed=seed_f)
        except RuntimeError:
            pass
        else:
            TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            rows.append(
                {
                    "split": split_i,
                    "method": "XGBoost",
                    "scope": "internal_test_B",
                    **met,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "TP": TP,
                }
            )
    return rows


def build_table_tex(df: pd.DataFrame, out_path: Path) -> None:
    metric_labels = [
        ("Accuracy", "acc"),
        ("Precision", "prec"),
        ("Recall", "rec"),
        ("Sensitivity", "sens"),
        ("Specificity", "spec"),
        ("F1 (class 1)", "f1"),
        ("AUC (ROC)", "auc"),
    ]
    display_names = {
        "LogReg-L2": "LR",
        "XGBoost": "XGB",
        "Concat": "Concat",
        "Single-Attn": "Single-Attn",
        "MTFusion": "MTFusion",
        "MTFusion+cross-attn": "MTFusion+cross",
    }
    method_order = [
        "LogReg-L2",
        "XGBoost",
        "Concat",
        "Single-Attn",
        "MTFusion",
        "MTFusion+cross-attn",
    ]

    lines: list[str] = []
    lines.append(r"\begin{tabular}{@{}llcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Metric} & \textbf{S1} & \textbf{S2} & \textbf{S3} & \textbf{S4} & \textbf{S5} & \textbf{Mean$\pm$SD} \\"
    )
    lines.append(r"\midrule")

    for method in method_order:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        disp = display_names.get(method, method)
        per_metric_folds: dict[str, list[float]] = {k: [] for _, k in metric_labels}
        for split_i in range(1, 6):
            srow = sub[sub["split"] == split_i]
            if srow.empty:
                continue
            r = srow.iloc[0]
            TN, FP, FN, TP = int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])
            n = TN + FP + FN + TP
            acc = (TP + TN) / max(1, n)
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            sens = rec
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            f1 = float(r["f1_class_1"])
            auc = float(r["auroc"])
            per_metric_folds["acc"].append(acc)
            per_metric_folds["prec"].append(prec)
            per_metric_folds["rec"].append(rec)
            per_metric_folds["sens"].append(sens)
            per_metric_folds["spec"].append(spec)
            per_metric_folds["f1"].append(f1)
            per_metric_folds["auc"].append(auc)

        for mi, (lab, key) in enumerate(metric_labels):
            vals = per_metric_folds[key]
            if len(vals) != 5:
                fold_cells = ["---"] * 5
                mean_cell = "---"
            else:
                fold_cells = [_fmt(v) for v in vals]
                mean_cell = _fmt_mean_std(vals)
            if mi == 0:
                prefix = rf"\multirow{{{len(metric_labels)}}}{{*}}{{\textbf{{{disp}}}}} & "
            else:
                prefix = "& "
            lines.append(
                prefix + f"{lab} & " + " & ".join(fold_cells) + f" & {mean_cell} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("results/cohortB_stagebi_5foldcv_deep.csv"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/cohortB_internal_metrics_table.tex"),
    )
    ap.add_argument("--cohort-b-npz", type=Path, default=Path("data/cohortB_stage_bi.npz"))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-classical", action="store_true")
    a = ap.parse_args()

    if a.csv.is_file():
        df_deep = pd.read_csv(a.csv)
    else:
        df_deep = pd.DataFrame()

    rows_classical: list[dict[str, object]] = []
    if not a.skip_classical and a.cohort_b_npz.is_file():
        rows_classical = classical_rows(a.cohort_b_npz, splits=a.splits, seed=a.seed)

    if rows_classical:
        df_c = pd.DataFrame(rows_classical)
        df = pd.concat([df_c, df_deep], ignore_index=True) if not df_deep.empty else df_c
    else:
        df = df_deep
    if df.empty:
        raise SystemExit("No data: provide --csv and/or --cohort-b-npz")

    build_table_tex(df, a.out)


if __name__ == "__main__":
    main()
