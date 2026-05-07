"""
prepare_dataset_npz.py
======================
Build a real `.npz` dataset compatible with `scripts/mtfusion.py` / training pipelines.

Inputs
------
- `ValidateSet-DKD.xlsx` (tabular variables + labels)
- `all_records/` (ECG sources; here: JSON with plotted pixels + WFDB-style .hea)

Output
------
An `.npz` with keys:
- ecg:    (N, n_ecg_features)   float32
- vitals: (N, n_vitals)         float32
- labels: (N,)                  int64  (zero-based)

This script supports three ECG feature sources:
1) `intervals` (recommended): extract clinically standard ECG intervals from WFDB waveforms
   (PR, QRS, QT, QTc, ST deviation, RR; in milliseconds).
2) `pixels` (fallback): compute simple geometry/slope features from JSON plotted pixels.
3) `image`: same per-lead pixel-geometry features as `pixels`, but traces are recovered from
   12-lead ECG printout JPG/PNG scans (default 3×4 lead layout).

The output format is compatible with MTFusion training code under `scripts/`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wfdb  # type: ignore
except Exception:  # pragma: no cover
    wfdb = None

try:
    import neurokit2 as nk  # type: ignore
except Exception:  # pragma: no cover
    nk = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


LEAD_ORDER: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

# Typical 12-lead report grid (3 rows × 4 columns), row-major cell order.
IMAGE_LEAD_GRID: tuple[tuple[str, str, str, str], ...] = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)


@dataclass(frozen=True)
class Config:
    xlsx_path: Path
    all_records_dir: Path
    ecg_image_dir: Path
    sheet: str
    label_col: str
    vitals_cols: tuple[str, ...]
    ecg_source: str
    ecg_lead: str
    out_path: Path
    # Crop fractions (top, bottom, left, right) applied before grid split for `image` source.
    image_crop: tuple[float, float, float, float]
    image_max_points_per_lead: int


def _robust_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _lead_features(pixels: np.ndarray) -> np.ndarray:
    """
    pixels: (N, 2) array of (x, y) plotted pixel coordinates.
    Returns a small numeric feature vector. All features are finite or NaN.
    """
    if pixels.size == 0:
        return np.full((10,), np.nan, dtype=np.float32)

    x = pixels[:, 0].astype(np.float32, copy=False)
    y = pixels[:, 1].astype(np.float32, copy=False)

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    x_rng = x_max - x_min
    y_rng = y_max - y_min

    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)

    # Slope stats after sorting by x (guarding against duplicate x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order]
    dx = np.diff(xs)
    dy = np.diff(ys)
    mask = dx != 0
    if np.any(mask):
        slopes = (dy[mask] / dx[mask]).astype(np.float32, copy=False)
        slope_abs_mean = float(np.mean(np.abs(slopes)))
        slope_abs_p95 = float(np.percentile(np.abs(slopes), 95))
    else:
        slope_abs_mean = float("nan")
        slope_abs_p95 = float("nan")

    n_points = float(len(x))
    return np.array(
        [
            n_points,
            x_min,
            x_max,
            x_rng,
            y_min,
            y_max,
            y_rng,
            y_mean,
            y_std,
            slope_abs_mean,
        ],
        dtype=np.float32,
    )


def _ecg_pixel_geometry_feature_names() -> list[str]:
    """Feature names for `_lead_features` stacked over `LEAD_ORDER` (same as JSON pixel path)."""
    names: list[str] = []
    for lead in LEAD_ORDER:
        names.extend(
            [
                f"{lead}.n_points",
                f"{lead}.x_min",
                f"{lead}.x_max",
                f"{lead}.x_range",
                f"{lead}.y_min",
                f"{lead}.y_max",
                f"{lead}.y_range",
                f"{lead}.y_mean",
                f"{lead}.y_std",
                f"{lead}.slope_abs_mean",
            ]
        )
    return names


def ecg_features_from_json(json_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
      - features: (len(LEAD_ORDER) * n_feat_per_lead,) float32
      - names:    same length list[str]
    If a lead appears multiple times, we average its feature vectors.
    """
    with json_path.open("r") as f:
        j = json.load(f)

    # Collect per-lead feature vectors (may have duplicates)
    per_lead: dict[str, list[np.ndarray]] = {k: [] for k in LEAD_ORDER}
    for ld in j.get("leads", []):
        name = ld.get("lead_name")
        if name not in per_lead:
            continue
        pixels = np.asarray(ld.get("plotted_pixels", []), dtype=np.float32)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            per_lead[name].append(np.full((10,), np.nan, dtype=np.float32))
        else:
            per_lead[name].append(_lead_features(pixels))

    feat_chunks: list[np.ndarray] = []
    for lead in LEAD_ORDER:
        vecs = per_lead[lead]
        if len(vecs) == 0:
            fvec = np.full((10,), np.nan, dtype=np.float32)
        elif len(vecs) == 1:
            fvec = vecs[0]
        else:
            fvec = np.nanmean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

        feat_chunks.append(fvec)

    feats = np.concatenate(feat_chunks, axis=0)
    return feats, _ecg_pixel_geometry_feature_names()


def _trace_pixels_from_gray_roi(
    gray_roi: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Recover (x, y) coordinates of dark ink in one lead panel (ROI pixel space).
    Uses a high percentile on inverted intensity, then subsamples for speed.
    """
    if gray_roi.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    inv = (255.0 - gray_roi.astype(np.float32)).ravel()
    inv2d = inv.reshape(gray_roi.shape)
    # Dark trace is bright in `inv2d`; take upper tail of the ROI histogram.
    for q in (93, 88, 82, 75):
        thr = float(np.percentile(inv2d, q))
        mask = inv2d >= thr
        ys, xs = np.where(mask)
        if xs.size > 0:
            break
    else:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def ecg_features_from_image(
    image_path: Path,
    *,
    crop: tuple[float, float, float, float] = (0.18, 0.08, 0.04, 0.04),
    max_points_per_lead: int = 8000,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the same 120-D pixel-geometry features as `ecg_features_from_json`, from a 12-lead
    ECG scan. Assumes `IMAGE_LEAD_GRID` matches the print layout.

    crop: (frac_top, frac_bottom, frac_left, frac_right) trimmed from each edge before gridding.
    """
    if Image is None:
        raise RuntimeError("ECG image features require `Pillow` (`pip install Pillow`).")

    names = _ecg_pixel_geometry_feature_names()
    rng = np.random.default_rng(seed)

    with Image.open(image_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)

    h, w = gray.shape
    t, b, lft, rgt = crop
    y0, y1 = int(t * h), int((1.0 - b) * h)
    x0, x1 = int(lft * w), int((1.0 - rgt) * w)
    y0, y1 = max(0, y0), max(y0 + 1, min(h, y1))
    x0, x1 = max(0, x0), max(x0 + 1, min(w, x1))
    crop_g = gray[y0:y1, x0:x1]
    ch, cw = crop_g.shape

    # Map lead name → pixels (merge duplicate cells if any)
    per_lead: dict[str, list[np.ndarray]] = {k: [] for k in LEAD_ORDER}
    n_rows, n_cols = len(IMAGE_LEAD_GRID), len(IMAGE_LEAD_GRID[0])
    rh, rw = ch // n_rows, cw // n_cols

    for ri, row_names in enumerate(IMAGE_LEAD_GRID):
        for ci, lead_name in enumerate(row_names):
            if lead_name not in per_lead:
                continue
            ys, ye = ri * rh, (ri + 1) * rh
            xs, xe = ci * rw, (ci + 1) * rw
            roi = crop_g[ys:ye, xs:xe]
            pts = _trace_pixels_from_gray_roi(roi, max_points_per_lead, rng)
            if pts.size == 0:
                per_lead[lead_name].append(np.full((10,), np.nan, dtype=np.float32))
            else:
                per_lead[lead_name].append(_lead_features(pts))

    feat_chunks: list[np.ndarray] = []
    for lead in LEAD_ORDER:
        vecs = per_lead[lead]
        if len(vecs) == 0:
            fvec = np.full((10,), np.nan, dtype=np.float32)
        elif len(vecs) == 1:
            fvec = vecs[0]
        else:
            fvec = np.nanmean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        feat_chunks.append(fvec)

    feats = np.concatenate(feat_chunks, axis=0)
    return feats, names


def _infer_ecg_image_path(row: pd.Series, image_dir: Path) -> Path | None:
    stem = str(row["ecg_filename"]).strip()
    if not stem:
        return None
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _infer_json_path(row: pd.Series, all_records_dir: Path) -> Path:
    # Excel has:
    #   ecg_filename: "45863011-0_0000"
    #   Unnamed: 1:   "p1028/p10281780/s45863011/45863011"
    # The `Unnamed: 1` column includes the record base-name as the last path
    # segment (e.g. ".../s45863011/45863011"), but the JSON lives directly
    # under the session directory (e.g. ".../s45863011/45863011-0_0000.json").
    rel = Path(str(row["Unnamed: 1"]).strip()).parent.as_posix()
    fname = str(row["ecg_filename"]).strip()
    return all_records_dir / rel / f"{fname}.json"


def _infer_wfdb_record_path(row: pd.Series, all_records_dir: Path) -> Path:
    # `Unnamed: 1` points to the WFDB record base path, e.g.
    # "p1028/p10281780/s45863011/45863011" (without extension).
    rel = Path(str(row["Unnamed: 1"]).strip())
    return all_records_dir / rel


def _safe_nanmedian(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    v = np.nanmedian(x)
    return float(v) if np.isfinite(v) else float("nan")


def ecg_interval_features_from_wfdb(
    record_path: Path,
    lead: str = "II",
) -> tuple[np.ndarray, list[str]]:
    """
    Extract interval/morphology features from a WFDB record.

    Returns:
      features: float32 (n_features,)
      names: list[str]

    Features (all in ms unless stated):
      - PR_ms (median across beats)
      - QRS_ms
      - QT_ms
      - RR_ms
      - QTc_bazett_ms
      - QTc_fridericia_ms
      - ST_dev_mV  (median ST deviation relative to PR baseline; mV)

    Notes:
      - Requires `wfdb` and `neurokit2`.
      - Uses a single lead for delineation (default II) for robustness.
      - On delineation failure, returns NaNs (later median-imputed).
    """
    names = [
        "PR_ms",
        "QRS_ms",
        "QT_ms",
        "RR_ms",
        "QTc_bazett_ms",
        "QTc_fridericia_ms",
        "ST_dev_mV",
    ]

    if wfdb is None or nk is None:
        raise RuntimeError(
            "ECG interval extraction requires `wfdb` and `neurokit2`. "
            "Install them or use --ecg-source=pixels."
        )

    rec = wfdb.rdrecord(str(record_path))
    fs = float(getattr(rec, "fs", np.nan))
    if not np.isfinite(fs) or fs <= 0:
        fs = 500.0  # conservative fallback; many of your headers are 500 Hz

    sig = rec.p_signal
    if sig is None:
        return np.full((len(names),), np.nan, dtype=np.float32), names

    sig = sig.astype(np.float32, copy=False)
    if sig.ndim != 2 or sig.shape[0] < int(0.5 * fs):
        return np.full((len(names),), np.nan, dtype=np.float32), names

    # Pick lead index
    lead_names = [str(s) for s in getattr(rec, "sig_name", [])]
    if lead_names and lead in lead_names:
        li = lead_names.index(lead)
    else:
        # fallback to channel 1 (often II) if available
        li = 1 if sig.shape[1] > 1 else 0

    x = sig[:, li]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    try:
        # Clean + R-peak detection
        signals, info = nk.ecg_process(x, sampling_rate=fs)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        if rpeaks.size < 3:
            return np.full((len(names),), np.nan, dtype=np.float32), names

        # Delineation (P/QRS/T boundaries)
        delineate, _ = nk.ecg_delineate(
            signals["ECG_Clean"].values if hasattr(signals["ECG_Clean"], "values") else signals["ECG_Clean"],
            rpeaks=rpeaks,
            sampling_rate=fs,
            method="dwt",
            show=False,
        )

        def _arr(key: str) -> np.ndarray:
            a = np.asarray(delineate.get(key, []), dtype=float)
            return a

        p_on = _arr("ECG_P_Onsets")
        qrs_on = _arr("ECG_QRS_Onsets")
        qrs_off = _arr("ECG_QRS_Offsets")
        t_off = _arr("ECG_T_Offsets")

        # Pairwise compute intervals where boundaries are finite
        def _ms(a: np.ndarray) -> np.ndarray:
            return (a / fs * 1000.0).astype(np.float32, copy=False)

        # PR: QRS_on - P_on
        pr = (qrs_on - p_on)
        pr = pr[np.isfinite(pr) & (pr > 0)]
        pr_ms = _safe_nanmedian(_ms(pr))

        # QRS duration: QRS_off - QRS_on
        qrs = (qrs_off - qrs_on)
        qrs = qrs[np.isfinite(qrs) & (qrs > 0)]
        qrs_ms = _safe_nanmedian(_ms(qrs))

        # QT: T_off - QRS_on
        qt = (t_off - qrs_on)
        qt = qt[np.isfinite(qt) & (qt > 0)]
        qt_ms_arr = _ms(qt)
        qt_ms = _safe_nanmedian(qt_ms_arr)

        # RR from R peaks
        rr = np.diff(rpeaks.astype(float))
        rr = rr[np.isfinite(rr) & (rr > 0)]
        rr_ms_arr = _ms(rr)
        rr_ms = _safe_nanmedian(rr_ms_arr)

        # QTc
        # QTc_Bazett = QT / sqrt(RR) (RR in seconds)
        rr_s = rr_ms_arr / 1000.0
        qt_s = qt_ms_arr / 1000.0
        if rr_s.size > 0 and qt_s.size > 0:
            # Align by using medians separately (robust) to avoid index mismatch
            rr_s_med = float(np.nanmedian(rr_s)) if np.isfinite(np.nanmedian(rr_s)) else float("nan")
            qt_s_med = float(np.nanmedian(qt_s)) if np.isfinite(np.nanmedian(qt_s)) else float("nan")
            if np.isfinite(rr_s_med) and rr_s_med > 0 and np.isfinite(qt_s_med):
                qtc_baz_ms = float(qt_s_med / math.sqrt(rr_s_med) * 1000.0)
                qtc_frd_ms = float(qt_s_med / (rr_s_med ** (1.0 / 3.0)) * 1000.0)
            else:
                qtc_baz_ms = float("nan")
                qtc_frd_ms = float("nan")
        else:
            qtc_baz_ms = float("nan")
            qtc_frd_ms = float("nan")

        # ST deviation (mV): sample at J + 60ms relative to PR baseline
        # baseline: mean between P_on and QRS_on (PR segment) for each beat
        st_devs = []
        j_offset = int(round(0.060 * fs))
        for po, qo, qf in zip(p_on, qrs_on, qrs_off):
            if not (np.isfinite(po) and np.isfinite(qo) and np.isfinite(qf)):
                continue
            po_i, qo_i, qf_i = int(po), int(qo), int(qf)
            if qo_i - po_i < int(0.02 * fs):  # too short
                continue
            base = float(np.mean(x[po_i:qo_i]))
            st_i = qf_i + j_offset
            if 0 <= st_i < x.shape[0]:
                st_devs.append(float(x[st_i] - base))
        st_dev_mV = float(np.nanmedian(st_devs)) if len(st_devs) else float("nan")

        feats = np.array(
            [pr_ms, qrs_ms, qt_ms, rr_ms, qtc_baz_ms, qtc_frd_ms, st_dev_mV],
            dtype=np.float32,
        )
        return feats, names
    except Exception:
        return np.full((len(names),), np.nan, dtype=np.float32), names


def _fit_imputer_medians(arr: np.ndarray) -> np.ndarray:
    """Column-wise median, ignoring NaNs (fallback to 0)."""
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.float32)


def _impute_inplace(arr: np.ndarray, medians: np.ndarray) -> None:
    nan_mask = ~np.isfinite(arr)
    if np.any(nan_mask):
        arr[nan_mask] = np.take(medians, np.where(nan_mask)[1])


def build_npz(cfg: Config) -> None:
    df = pd.read_excel(cfg.xlsx_path, sheet_name=cfg.sheet)

    # Keep rows that have an ECG filename and a label
    df = df[df["ecg_filename"].notna()]
    if cfg.ecg_source in ("intervals", "pixels"):
        if "Unnamed: 1" not in df.columns:
            raise RuntimeError(
                "Excel is missing column `Unnamed: 1` (record path). "
                "Use --ecg-source=image for image-only workbooks, or add the path column."
            )
        df = df[df["Unnamed: 1"].notna()]
    df = df[df[cfg.label_col].notna()].copy()

    # Labels → zero-based contiguous ints
    raw_labels = df[cfg.label_col].map(_robust_float).to_numpy()
    uniq = np.unique(raw_labels[~np.isnan(raw_labels)])
    uniq_sorted = np.sort(uniq)
    mapping = {float(v): i for i, v in enumerate(uniq_sorted.tolist())}
    y = np.array([mapping[float(v)] for v in raw_labels], dtype=np.int64)

    # Vitals (tabular)
    vitals_df = df.loc[:, list(cfg.vitals_cols)].copy()
    for c in vitals_df.columns:
        vitals_df[c] = vitals_df[c].map(_robust_float)
    X_vitals = vitals_df.to_numpy(dtype=np.float32)

    # ECG features
    ecg_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    keep_mask: list[bool] = []

    for _, row in df.iterrows():
        if cfg.ecg_source == "intervals":
            rp = _infer_wfdb_record_path(row, cfg.all_records_dir)
            if not (rp.with_suffix(".hea").exists() or rp.exists()):
                keep_mask.append(False)
                continue
            feats, names = ecg_interval_features_from_wfdb(rp, lead=cfg.ecg_lead)
        elif cfg.ecg_source == "image":
            ip = _infer_ecg_image_path(row, cfg.ecg_image_dir)
            if ip is None:
                keep_mask.append(False)
                continue
            feats, names = ecg_features_from_image(
                ip,
                crop=cfg.image_crop,
                max_points_per_lead=cfg.image_max_points_per_lead,
            )
        else:
            jp = _infer_json_path(row, cfg.all_records_dir)
            if not jp.exists():
                keep_mask.append(False)
                continue
            feats, names = ecg_features_from_json(jp)
        feature_names = feature_names or names
        ecg_rows.append(feats)
        keep_mask.append(True)

    keep = np.array(keep_mask, dtype=bool)
    if keep.sum() == 0:
        raise RuntimeError(
            f"No ECG records resolved for ecg_source={cfg.ecg_source!r} (check paths / filenames)."
        )

    X_ecg = np.stack(ecg_rows, axis=0).astype(np.float32)
    X_vitals = X_vitals[keep].astype(np.float32)
    y = y[keep]

    # Simple NaN imputation (median per column)
    ecg_med = _fit_imputer_medians(X_ecg)
    vit_med = _fit_imputer_medians(X_vitals)
    _impute_inplace(X_ecg, ecg_med)
    _impute_inplace(X_vitals, vit_med)

    out = cfg.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ecg=X_ecg,
        vitals=X_vitals,
        labels=y,
        ecg_feature_names=np.array(feature_names or [], dtype=object),
        vital_feature_names=np.array(list(cfg.vitals_cols), dtype=object),
        label_values=np.array(sorted(set(mapping.keys())), dtype=np.float32),
        label_col=np.array([cfg.label_col], dtype=object),
    )

    print(f"Wrote: {out}")
    print(f"X_ecg: {X_ecg.shape}  X_vitals: {X_vitals.shape}  y: {y.shape}")
    print(f"Classes (raw {cfg.label_col} values): {sorted(set(mapping.keys()))}")
    print(f"Dropped rows (missing ECG source file): {int((~keep).sum())} / {len(keep)}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Create .npz dataset for MTFusion / fusion experiments")
    p.add_argument(
        "--xlsx",
        type=Path,
        default=Path("ValidateSet-DKD.xlsx"),
        help="Path to ValidateSet-DKD.xlsx",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default="Parameters",
        help="Sheet to read from the Excel workbook",
    )
    p.add_argument(
        "--all-records",
        type=Path,
        default=Path("all_records"),
        help="Root directory containing the all_records tree",
    )
    p.add_argument(
        "--ecg-image-dir",
        type=Path,
        default=Path("ProcessedData"),
        help="Folder with ECG report images when using --ecg-source=image",
    )
    p.add_argument(
        "--image-crop",
        type=str,
        default="0.18,0.08,0.04,0.04",
        help="For image source: fractional crop top,bottom,left,right before 12-lead grid split",
    )
    p.add_argument(
        "--image-max-points",
        type=int,
        default=8000,
        help="Max foreground pixels sampled per lead panel (image source)",
    )
    p.add_argument(
        "--label-col",
        type=str,
        default="stage_multi",
        help="Label column in the Excel sheet (e.g. stage_multi or stage_bi)",
    )
    p.add_argument(
        "--vitals",
        type=str,
        default="age,gender,pulse",
        help="Comma-separated vitals/tabular columns to use",
    )
    p.add_argument(
        "--ecg-source",
        type=str,
        default="intervals",
        choices=["intervals", "pixels", "image"],
        help="ECG feature source: intervals (WFDB), pixels (JSON traces), or image (scanned 12-lead)",
    )
    p.add_argument(
        "--ecg-lead",
        type=str,
        default="II",
        help="Lead name to use for interval extraction (default: II)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/validate_dkd.npz"),
        help="Output .npz path",
    )
    a = p.parse_args()

    parts = [float(x.strip()) for x in a.image_crop.split(",")]
    if len(parts) != 4:
        raise SystemExit("--image-crop must be four comma-separated floats: top,bottom,left,right")
    image_crop = (parts[0], parts[1], parts[2], parts[3])

    vitals_cols = tuple([c.strip() for c in a.vitals.split(",") if c.strip()])
    return Config(
        xlsx_path=a.xlsx,
        all_records_dir=a.all_records,
        ecg_image_dir=a.ecg_image_dir,
        sheet=a.sheet,
        label_col=a.label_col,
        vitals_cols=vitals_cols,
        ecg_source=a.ecg_source,
        ecg_lead=a.ecg_lead,
        out_path=a.out,
        image_crop=image_crop,
        image_max_points_per_lead=a.image_max_points,
    )


def main() -> None:
    cfg = parse_args()
    build_npz(cfg)


if __name__ == "__main__":
    main()

