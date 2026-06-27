#!/usr/bin/env python3
"""
build_notebook.py — Build presentation.ipynb from narrative markdown + result CSVs.

Run from repo root:
    python source/reporting/build_notebook.py

Output: presentation/presentation.ipynb
"""
import sys
import subprocess
import inspect
from pathlib import Path

import nbformat as nbf
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.parent
RESULTS_DIR  = REPO_ROOT / "results"
NARR_DIR     = REPO_ROOT / "source" / "reporting" / "results"
PHASE_DIR    = RESULTS_DIR / "deep_res_mlp_results"
OUT_DIR      = REPO_ROOT / "presentation"
OUT_PATH     = OUT_DIR / "presentation.ipynb"

# Columns that must exist in each CSV (build-time assertion)
REQUIRED_COLUMNS = {
    "eda_pca.csv":              ["tau", "label", "pca1", "pca2"],
    "eda_homophily.csv":        ["tau", "licit_licit", "illicit_illicit",
                                  "illicit_licit", "illicit_unknown"],
    "eda_drift.csv":            ["tau", "mmd", "wasserstein_pca"],
    "snapshot_topology.csv":    ["Tau", "N_illicit", "N_licit", "Illicit_Rate"],
    "sweep_results.csv":        ["Sweep", "Static_Time_s",
                                  "Static_OOT_Pooled_PRAUC", "Static_OOT_Pooled_F1"],
    "final_aggregated_results.csv": ["Sweep", "Variation",
                                      "Static_OOT_Pooled_PRAUC_mean",
                                      "Static_OOT_Pooled_PRAUC_std"],
    "walk_forward_timesteps.csv": ["Sweep", "Seed", "Tau", "N_illicit",
                                    "Low_Confidence", "Regime", "F1", "PRAUC"],
}

# Sweep strings referenced directly in code cells (validated in pre-flight)
SWEEP_LITERALS = {
    "walk_forward_timesteps.csv": [
        "Baseline: XGBoost WF (epsilon-fallback)",
        "Best WF: Sweep 1: SGC (baseline) (Seed 42, Var Base)",
        "Best WF: Sweep 2: + MLP Head (Seed 42, Var Base)",
        "Best WF: Grid: K=2, Dir=T, Topo=early (Seed 42, Var Base)",
    ],
}


# ── PRE-FLIGHT ───────────────────────────────────────────────────────────────
def _preflight():
    errors = []

    # 1. Check all required CSVs exist and have expected columns
    for fname, cols in REQUIRED_COLUMNS.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            errors.append(f"MISSING CSV: {path}")
            continue
        df = pd.read_csv(path, nrows=1)
        for col in cols:
            if col not in df.columns:
                errors.append(f"MISSING COLUMN: {col!r} in {fname}")

    # 2. Check all Sweep literals exist in the real CSV data
    for fname, literals in SWEEP_LITERALS.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            continue  # already caught above
        real_sweeps = set(pd.read_csv(path)["Sweep"].unique())
        for lit in literals:
            if lit not in real_sweeps:
                errors.append(f"SWEEP LITERAL NOT IN CSV: {lit!r} (in {fname})")

    # 3. Check narrative markdown files exist
    for md_file in [
        "eda_embeddings_analysis.md", "eda_homophily_analysis.md",
        "diagnostic_falsification_report.md", "tda.md",
        "baseline_performance_report.md", "sgc_grid_analysis.md",
        "deep_res_mlp_analysis.md", "wf_temporal_analysis.md",
    ]:
        if not (NARR_DIR / md_file).exists():
            errors.append(f"MISSING NARRATIVE: {NARR_DIR / md_file}")

    # 4. Check phase aggregated CSVs for Section 5
    for ph in ["A", "B", "C", "D"]:
        p = PHASE_DIR / f"sweep_phase{ph}" / f"phase{ph}_aggregated.csv"
        if not p.exists():
            errors.append(f"MISSING PHASE CSV: {p}")

    if errors:
        for e in errors:
            print(f"[PRE-FLIGHT ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("[PRE-FLIGHT] All checks passed.")
