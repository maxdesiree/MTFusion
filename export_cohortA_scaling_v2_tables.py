"""
Build human-readable tables from ``cohortA_scaling_summary.csv``.

Writes (defaults under ``results/cohortA_scaling_v2/``):

- ``cohortA_scaling_table.md``
- ``cohortA_scaling_table.tex``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        type=Path,
        default=_ROOT / "results" / "cohortA_scaling_v2" / "cohortA_scaling_summary.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "results" / "cohortA_scaling_v2")
    a = ap.parse_args()

    df = pd.read_csv(a.summary)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = df.sort_values(["Dataset size", "Model"], kind="stable")
    md_lines = [
        "| Dataset size | Model | Micro-F1 | Macro PR-AUC | Macro-F1 |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, r in d.iterrows():
        md_lines.append(
            f"| {int(r['Dataset size'])} | {r['Model']} | {r['Micro-F1']} | {r['Macro PR-AUC']} | {r['Macro-F1']} |"
        )
    md_path = out_dir / "cohortA_scaling_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{rlccc}",
        r"\hline",
        r"\textbf{N} & \textbf{Model} & \textbf{Micro-F1} & \textbf{Macro PR-AUC} & \textbf{Macro-F1} \\",
        r"\hline",
    ]
    for _, r in d.iterrows():
        tex_lines.append(
            f"{int(r['Dataset size'])} & {r['Model']} & {r['Micro-F1']} & {r['Macro PR-AUC']} & {r['Macro-F1']} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}", ""])
    tex_path = out_dir / "cohortA_scaling_table.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
