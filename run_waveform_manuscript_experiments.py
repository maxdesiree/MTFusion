#!/usr/bin/env python3
"""
Backward-compatible entrypoint.

The cohort naming in this release is:
- Cohort A: N=353
- Cohort B: N=54

The waveform (WFDB) pipeline is Cohort B, and the main implementation is now
``scripts/run_cohortB_waveform.py``.
"""

from __future__ import annotations

from scripts.run_cohortB_waveform import main


if __name__ == "__main__":
    main()

