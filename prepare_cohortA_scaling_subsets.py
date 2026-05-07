"""
Build cohort A (N=353) scaling subsets (Steps 1 & 3).

1) Iterative multilabel stratification: pool N=10,000 from the full cohort, then N=1,500.
3) From dataset_1,500, draw N=500 and N=1,000 with MultilabelStratifiedShuffleSplit for seeds [42,123,999].

Outputs under ``data/cohortA_scaling_v2/`` (override with ``--out-dir``):
  - cohortA_pool_10000.csv
  - dataset_1500.csv
  - dataset_500_seed1.csv … dataset_1000_seed3.csv
  - manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cohortA_data import encode_gender_series, rhythm_one_hot, tabular_columns


def _label_counts_df(df: pd.DataFrame) -> dict[str, int]:
    return df["Rhythm"].astype(str).value_counts().sort_index().to_dict()


def _rare_class_priority_scores(y_rows: np.ndarray) -> np.ndarray:
    """Higher score ⇒ higher priority to keep when trimming (rarer active class in this pool)."""
    counts = np.maximum(y_rows.sum(axis=0), 1.0)
    cls = y_rows.argmax(axis=1)
    return 1.0 / counts[cls]


def _adjust_train_indices(
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    *,
    n_keep: int,
    seed: int,
) -> np.ndarray:
    """
    ``MultilabelStratifiedShuffleSplit`` sometimes returns |tr| != ``n_keep`` (library rounding).
    Trim or borrow using a deterministic rare-class priority rule on label counts in the pool.
    """
    rng = np.random.default_rng(int(seed) + 91_337)
    tr = np.asarray(tr, dtype=int)
    te = np.asarray(te, dtype=int)
    if len(tr) == n_keep:
        return tr
    if len(tr) > n_keep:
        y_tr = y[tr]
        prio = _rare_class_priority_scores(y_tr)
        tie = rng.random(len(tr))
        order = np.lexsort((tie, -prio))
        keep_local = order[:n_keep]
        return tr[keep_local]

    need = n_keep - len(tr)
    if need <= 0 or len(te) < need:
        raise RuntimeError(f"Cannot grow train from |tr|={len(tr)} to {n_keep} (|te|={len(te)}).")
    y_tr = y[tr]
    # Prefer held-out rows whose class is underrepresented in ``tr`` vs prevalence in ``tr ∪ te``.
    pool = np.concatenate([tr, te], axis=0)
    y_pool = y[pool]
    target = y_pool.sum(axis=0) / max(1, len(pool))
    cur = y_tr.sum(axis=0) / max(1, len(tr))
    deficit = target - cur

    y_te = y[te]
    cls_te = y_te.argmax(axis=1)
    add_prio = deficit[cls_te]
    tie = rng.random(len(te))
    order_te = np.lexsort((tie, -add_prio))
    pick = te[order_te[:need]]
    return np.concatenate([tr, pick], axis=0)


def _split_stratified_subset(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    n_keep: int,
    seed: int,
) -> pd.DataFrame:
    n = len(df)
    if n_keep > n:
        raise ValueError(f"Requested n_keep={n_keep} but only n={n}.")
    if n_keep == n:
        return df.reset_index(drop=True)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=n_keep,
        test_size=n - n_keep,
        random_state=int(seed),
    )
    tr, te = next(msss.split(np.zeros((n, 1)), y))
    tr = _adjust_train_indices(y, tr, te, n_keep=n_keep, seed=int(seed))
    return df.iloc[tr].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=_ROOT / "cohortA_Diagnostics.xlsx")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "data" / "cohortA_scaling_v2")
    ap.add_argument("--pool-n", type=int, default=10_000)
    ap.add_argument("--base-n", type=int, default=1_500)
    ap.add_argument("--pool-seed", type=int, default=42)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--sizes", type=str, default="500,1000")
    ap.add_argument("--subset-seeds", type=str, default="42,123,999")
    a = ap.parse_args()

    df = pd.read_excel(a.xlsx)
    df = df[df["FileName"].notna() & df["Rhythm"].notna()].copy()
    df["Gender"] = encode_gender_series(df["Gender"])
    tab_cols = tabular_columns(df)
    for c in tab_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    y_full = rhythm_one_hot(df["Rhythm"])
    n_full = len(df)
    pool_n = min(int(a.pool_n), n_full)
    if pool_n != int(a.pool_n):
        print(f"Note: --pool-n={a.pool_n} capped to available rows {n_full}.")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pool = _split_stratified_subset(df, y_full, n_keep=pool_n, seed=a.pool_seed)
    pool_path = out_dir / f"cohortA_pool_{pool_n}.csv"
    df_pool.to_csv(pool_path, index=False)
    print(f"Wrote pool ({pool_n}): {pool_path}")

    y_pool = rhythm_one_hot(df_pool["Rhythm"])
    df_base = _split_stratified_subset(df_pool, y_pool, n_keep=int(a.base_n), seed=a.base_seed)
    base_path = out_dir / f"dataset_{int(a.base_n)}.csv"
    df_base.to_csv(base_path, index=False)
    print(f"Wrote dataset_{int(a.base_n)} ({len(df_base)}): {base_path}")

    sizes = [int(x.strip()) for x in a.sizes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in a.subset_seeds.split(",") if x.strip()]
    y_base = rhythm_one_hot(df_base["Rhythm"])
    n_base = len(df_base)

    manifest: dict = {
        "pool_n": pool_n,
        "pool_seed": int(a.pool_seed),
        "base_n": int(a.base_n),
        "base_seed": int(a.base_seed),
        "files": {
            f"cohortA_pool_{pool_n}.csv": {"n": len(df_pool), "rhythm_counts": _label_counts_df(df_pool)},
            f"dataset_{int(a.base_n)}.csv": {"n": len(df_base), "rhythm_counts": _label_counts_df(df_base)},
        },
        "subsets": [],
    }

    for sz in sizes:
        for sj, seed in enumerate(seeds, start=1):
            if sz > n_base:
                raise SystemExit(f"Subset size {sz} exceeds dataset_{int(a.base_n)} n={n_base}.")
            sub = _split_stratified_subset(df_base, y_base, n_keep=sz, seed=seed)
            fname = f"dataset_{sz}_seed{sj}.csv"
            sub.to_csv(out_dir / fname, index=False)
            manifest["subsets"].append(
                {
                    "path": fname,
                    "n": len(sub),
                    "target_n": sz,
                    "seed": seed,
                    "rhythm_counts": _label_counts_df(sub),
                }
            )
            print(f"Wrote {fname} (n={len(sub)}, seed={seed})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
