# Delaying Modality Compression with Multi-Token Fusion for Limited Physiological Datasets

This repository contains the **model code and experiment runners**.
The manuscript is submitted to the ICONIP 2026, Melbourne, Australia.
It is intentionally **code-only** for GitHub reproducibility.

## What’s in here

- `scripts/`: training/evaluation entrypoints and model definitions (PyTorch + sklearn baselines).
- `repro/`: no-data reproducibility helpers (imports + version printout). private dataset available upon reseanable request.

## Environment setup

Create a fresh environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```
Sanity check (no data required):

```bash
python repro/smoke_imports.py
python repro/print_versions.py
```

## Running the main pipelines (data required)
### Cohort A (imaging report)

Subset generation + scaling experiment runner:

```bash
python scripts/prepare_cohortA_scaling_subsets.py --help
python scripts/run_cohortA.py --help
```

### Cohort B (digital waveforms)

Entry point:

```bash
python scripts/run_cohortB_waveform.py --help
```

Typical k-fold CV run (expects `ValidateSet-DKD.xlsx` and `all_records/` present locally):

```bash
python scripts/run_cohortB_waveform.py \
  --xlsx ValidateSet-DKD.xlsx \
  --sheet Parameters \
  --all-records all_records \
  --label-col stage_bi \
  --tab-cols age,gender,pulse \
  --folds 5 \
  --epochs 50 \
  --device cpu
```

Outputs go under `results/cohortB_waveform/` by default.

### Cohort B (image ECG scans + tabular fusion)

Entry point:

```bash
python scripts/run_cohortB_stagebi_5splits_externalA.py --help
```

Typical run (expects `data/cohortB_stage_bi.npz`, `ValidateSet-ImageDKD.xlsx`, and `ProcessedData/`):

```bash
python scripts/run_cohortB_stagebi_5splits_externalA.py \
  --cohort-b data/cohortB_stage_bi.npz \
  --xlsx ValidateSet-ImageDKD.xlsx \
  --sheet Parameters \
  --image-dir ProcessedData \
  --splits 5 \
  --device cpu
```

