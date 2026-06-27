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


# ── IMPORT PARSER (for build-time filter validation) ────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from sweep_parser import parse_sweep, select, add_parsed_columns, family_config_id


# ── CELL BUILDERS ────────────────────────────────────────────────────────────

def _md(source: str) -> nbf.NotebookNode:
    """Return a markdown cell."""
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    """Return a code cell."""
    return nbf.v4.new_code_cell(source)


def _read_narrative(filename: str) -> str:
    """Read a narrative markdown file verbatim."""
    return (NARR_DIR / filename).read_text(encoding="utf-8")


def build_boilerplate_cells() -> list:
    """Return the three mandatory preamble cells for Colab execution."""

    # Cell 0 — pip installs (giotto-tda before torch to avoid conflicts)
    cell0 = _code(
        "# Install dependencies\n"
        "!pip install --quiet nbformat pandas matplotlib seaborn\n"
        "# Note: giotto-tda not needed at runtime — TDA was a build-time analysis\n"
        "print('Dependencies ready.')"
    )

    # Cell 1 — Drive mount with local fallback + path config
    cell1 = _code(
        "import os\n"
        "from pathlib import Path\n\n"
        "try:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    RESULTS_DIR = '/content/drive/MyDrive/elliptic/results/'\n"
        "except ImportError:\n"
        "    # Running locally\n"
        "    RESULTS_DIR = str(Path().resolve() / 'results') + '/'\n\n"
        "print(f'RESULTS_DIR = {RESULTS_DIR}')\n"
        "# Verify key CSV is accessible\n"
        "assert os.path.exists(RESULTS_DIR + 'sweep_results.csv'), \\\n"
        "    f'Cannot find results at {RESULTS_DIR} — check Drive path or local path'"
    )

    # Cell 2 — sweep_parser source embedded verbatim (no Drive dependency)
    parser_source = (Path(__file__).parent / "sweep_parser.py").read_text(encoding="utf-8")
    cell2 = _code(
        "# sweep_parser — auto-embedded by build_notebook.py\n"
        + parser_source
        + "\nprint('sweep_parser loaded.')"
    )

    return [cell0, cell1, cell2]


def build_section_1() -> list:
    """EDA: PCA embeddings + homophily over time."""
    cells = []
    cells.append(_md(
        "---\n"
        + _read_narrative("eda_embeddings_analysis.md")
    ))
    cells.append(_code(
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "df_pca = pd.read_csv(f'{RESULTS_DIR}eda_pca.csv')\n"
        "color_map = {0: '#2ecc71', 1: '#e74c3c', -1: '#95a5a6'}\n"
        "label_map = {0: 'Licit', 1: 'Illicit', -1: 'Unknown'}\n\n"
        "fig, ax = plt.subplots(figsize=(10, 7))\n"
        "for lv in [-1, 0, 1]:\n"
        "    sub = df_pca[df_pca['label'] == lv]\n"
        "    ax.scatter(sub['pca1'], sub['pca2'],\n"
        "               c=color_map[lv], label=label_map[lv],\n"
        "               alpha=0.3, s=10, rasterized=True)\n"
        "ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')\n"
        "ax.set_title('PCA Embedding of Elliptic Dataset (166 Features)')\n"
        "ax.legend(markerscale=3, framealpha=0.9)\n"
        "plt.tight_layout(); plt.show()"
    ))
    cells.append(_md(_read_narrative("eda_homophily_analysis.md")))
    cells.append(_code(
        "df_h = pd.read_csv(f'{RESULTS_DIR}eda_homophily.csv')\n"
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "cols_colors = {\n"
        "    'licit_licit':     '#2ecc71',\n"
        "    'illicit_illicit': '#e74c3c',\n"
        "    'illicit_licit':   '#e67e22',\n"
        "    'illicit_unknown': '#9b59b6',\n"
        "}\n"
        "for col, color in cols_colors.items():\n"
        "    ax.plot(df_h['tau'], df_h[col],\n"
        "            label=col.replace('_', '–'), color=color, linewidth=2)\n"
        "ax.axvline(43, color='red', linestyle='--', alpha=0.7, label='τ=43 (AlphaBay)')\n"
        "ax.set_xlabel('Time Step τ'); ax.set_ylabel('Edge Count')\n"
        "ax.set_title('Homophily: Edge-Type Counts Over Time')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ))
    return cells


def build_section_2() -> list:
    """τ=43 Anomaly: drift + prevalence collapse."""
    cells = []
    cells.append(_md(
        "---\n"
        + _read_narrative("diagnostic_falsification_report.md")
    ))
    # Plot 2a: MMD + Wasserstein drift (dual-axis)
    cells.append(_code(
        "df_drift = pd.read_csv(f'{RESULTS_DIR}eda_drift.csv')\n"
        "fig, ax1 = plt.subplots(figsize=(12, 5))\n"
        "ax2 = ax1.twinx()\n"
        "ax1.plot(df_drift['tau'], df_drift['mmd'],\n"
        "         color='#3498db', linewidth=2, label='MMD (Feature Drift)')\n"
        "ax2.plot(df_drift['tau'], df_drift['wasserstein_pca'],\n"
        "         color='#e74c3c', linewidth=2, linestyle='--',\n"
        "         label='Wasserstein-PCA (Embedding Drift)')\n"
        "for tau_val, color, label in [(43, 'red', 'τ=43'), (44, 'orange', 'τ=44')]:\n"
        "    ax1.axvline(tau_val, color=color, linestyle=':', linewidth=2, alpha=0.8)\n"
        "    ax1.text(tau_val + 0.3, ax1.get_ylim()[1] * 0.85,\n"
        "             label, color=color, fontsize=9)\n"
        "ax1.set_xlabel('Time Step τ')\n"
        "ax1.set_ylabel('MMD', color='#3498db')\n"
        "ax2.set_ylabel('Wasserstein (PCA)', color='#e74c3c')\n"
        "ax1.set_title('Covariate Drift Over Time — Spike Occurs AFTER τ=43')\n"
        "l1, lb1 = ax1.get_legend_handles_labels()\n"
        "l2, lb2 = ax2.get_legend_handles_labels()\n"
        "ax1.legend(l1 + l2, lb1 + lb2, loc='upper left')\n"
        "plt.tight_layout(); plt.show()"
    ))
    cells.append(_md(_read_narrative("tda.md")))
    # Plot 2b: N_illicit prevalence (full 49-step series from snapshot_topology)
    cells.append(_code(
        "df_topo = pd.read_csv(f'{RESULTS_DIR}snapshot_topology.csv')\n"
        "bar_colors = ['#e74c3c' if t == 43 else '#3498db'\n"
        "              for t in df_topo['Tau']]\n"
        "fig, ax = plt.subplots(figsize=(14, 5))\n"
        "ax.bar(df_topo['Tau'], df_topo['N_illicit'],\n"
        "       color=bar_colors, alpha=0.85)\n"
        "ax.axvline(43, color='red', linestyle='--', alpha=0.5)\n"
        "ax.annotate('τ=43\\n(AlphaBay\\nshutdown)',\n"
        "            xy=(43, df_topo.loc[df_topo['Tau']==43, 'N_illicit'].values[0]),\n"
        "            xytext=(40, 150), fontsize=9, color='red',\n"
        "            arrowprops=dict(arrowstyle='->', color='red'))\n"
        "ax.set_xlabel('Time Step τ')\n"
        "ax.set_ylabel('Number of Illicit Nodes')\n"
        "ax.set_title('Illicit Node Count — 90% Collapse at τ=43 (Prior Probability Shift)')\n"
        "plt.tight_layout(); plt.show()"
    ))
    return cells


def build_section_3() -> list:
    """Tabular baselines vs graph models: Training Time vs OOT Pooled PRAUC."""
    cells = [_md("---\n" + _read_narrative("baseline_performance_report.md"))]
    cells.append(_code(
        "import numpy as np\n\n"
        "df_sr = pd.read_csv(f'{RESULTS_DIR}sweep_results.csv')\n"
        "df_sr = add_parsed_columns(df_sr)\n"
        "df_sr['_config_id'] = df_sr['Sweep'].map(family_config_id)\n\n"
        "# Aggregate by config (collapses seeds and Base/PCA variants)\n"
        "agg = (\n"
        "    df_sr\n"
        "    .dropna(subset=['Static_Time_s', 'Static_OOT_Pooled_PRAUC'])\n"
        "    .groupby('_family')\n"
        "    .agg(\n"
        "        time_mean=('Static_Time_s', 'mean'),\n"
        "        prauc_mean=('Static_OOT_Pooled_PRAUC', 'mean'),\n"
        "        prauc_std=('Static_OOT_Pooled_PRAUC', 'std'),\n"
        "    )\n"
        "    .reset_index()\n"
        "    .rename(columns={'_family': 'family'})\n"
        "    .dropna(subset=['prauc_mean'])\n"
        ")\n"
        "# Exclude IsolationForest (PRAUC is NaN/noise under anomaly scoring)\n"
        "agg = agg[agg['family'] != 'IsolationForest']\n\n"
        "palette = {\n"
        "    'XGBoost': '#e74c3c', 'RandomForest': '#e67e22',\n"
        "    'SGC+MLP': '#3498db', 'SGC': '#9b59b6',\n"
        "    'LogisticRegression': '#1abc9c', 'GCN': '#34495e',\n"
        "}\n"
        "display_names = {\n"
        "    'LogisticRegression': 'Logistic Reg.', 'GCN': 'PyG GCN',\n"
        "}\n\n"
        "fig, ax = plt.subplots(figsize=(11, 6))\n"
        "for _, row in agg.iterrows():\n"
        "    color = palette.get(row['family'], '#7f8c8d')\n"
        "    ax.scatter(row['time_mean'], row['prauc_mean'],\n"
        "               color=color, s=180, zorder=5)\n"
        "    name = display_names.get(row['family'], row['family'])\n"
        "    ax.annotate(name, (row['time_mean'], row['prauc_mean']),\n"
        "                textcoords='offset points', xytext=(8, 4), fontsize=10)\n"
        "    if pd.notna(row['prauc_std']) and row['prauc_std'] > 0:\n"
        "        ax.errorbar(row['time_mean'], row['prauc_mean'],\n"
        "                    yerr=row['prauc_std'], fmt='none',\n"
        "                    color='grey', capsize=4, alpha=0.6)\n"
        "ax.set_xscale('log')\n"
        "ax.set_xlabel('Training Time (seconds, log scale)')\n"
        "ax.set_ylabel('OOT Pooled PRAUC  [primary metric]')\n"
        "ax.set_title('Computational Cost vs. OOT Performance\\n'\n"
        "             'Error bars = ±1 std across 3 seeds (SGC/SGC+MLP)')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ))
    return cells


def build_section_4() -> list:
    """SGC+MLP grid search: K depth × PCA — the oversmoothing + savior story."""
    cells = [_md("---\n" + _read_narrative("sgc_grid_analysis.md"))]
    cells.append(_code(
        "df_fa = pd.read_csv(f'{RESULTS_DIR}final_aggregated_results.csv')\n"
        "df_fa = add_parsed_columns(df_fa)\n\n"
        "# Grid rows only — explicit K=/Dir=/Topo= strings\n"
        "grid = df_fa[\n"
        "    select(df_fa, family_tag='Grid')\n"
        "    & df_fa['_K'].notna()\n"
        "    & df_fa['_variation'].notna()\n"
        "].copy()\n"
        "grid['K'] = grid['_K'].astype(int)\n"
        "grid['Var'] = grid['_variation']\n\n"
        "# Best PRAUC per (K, Var) combination\n"
        "pivot = (\n"
        "    grid\n"
        "    .groupby(['K', 'Var'])['Static_OOT_Pooled_PRAUC_mean']\n"
        "    .max()\n"
        "    .unstack('Var')\n"
        ")\n\n"
        "k_vals = [1, 2, 3]\n"
        "width = 0.35\n"
        "base_vals = [pivot.get('Base', {}).get(k, 0) for k in k_vals]\n"
        "pca_vals  = [pivot.get('PCA',  {}).get(k, 0) for k in k_vals]\n\n"
        "fig, ax = plt.subplots(figsize=(9, 6))\n"
        "xs = list(range(len(k_vals)))\n"
        "ax.bar([x - width/2 for x in xs], base_vals, width,\n"
        "       label='Raw (Base)', color='#e74c3c', alpha=0.85)\n"
        "ax.bar([x + width/2 for x in xs], pca_vals,  width,\n"
        "       label='PCA',        color='#3498db', alpha=0.85)\n"
        "ax.set_xlabel('Neighborhood Depth K')\n"
        "ax.set_ylabel('Best OOT Pooled PRAUC  [primary metric]')\n"
        "ax.set_title(\n"
        "    'PCA as Oversmoothing Regularizer\\n'\n"
        "    'K=3 Raw → collapse; K=3 PCA → best graph-model OOT score'\n"
        ")\n"
        "ax.set_xticks(xs); ax.set_xticklabels(['K=1', 'K=2', 'K=3'])\n"
        "ax.legend()\n"
        "# Annotation: oversmoothing collapse arrow\n"
        "if len(base_vals) >= 3 and base_vals[2] > 0:\n"
        "    ax.annotate(\n"
        "        'Oversmoothing collapse',\n"
        "        xy=(2 - width/2, base_vals[2]),\n"
        "        xytext=(1.2, base_vals[2] + 0.04),\n"
        "        fontsize=9, color='#e74c3c',\n"
        "        arrowprops=dict(arrowstyle='->', color='#e74c3c'),\n"
        "    )\n"
        "plt.tight_layout(); plt.show()\n\n"
        "# Disambiguation note\n"
        "print('NOTE: PCA here = input-compression regularizer (reduces oversmoothing at K=3).')\n"
        "print('This is distinct from the drift-diagnostic PCA in Section 2.')"
    ))
    return cells


# ── SECTION 5 ─────────────────────────────────────────────────────────────────

def _build_phase_table() -> str:
    """Read phaseA-D aggregated CSVs at build time. Returns markdown table string."""
    rows = []

    # Phase A: best by OOT_Pooled_PRAUC_mean
    df_a = pd.read_csv(PHASE_DIR / "sweep_phaseA" / "phaseA_aggregated.csv")
    best_a = df_a.sort_values("OOT_Pooled_PRAUC_mean", ascending=False).iloc[0]
    rows.append(
        f"| A: Architecture depth | K ∈ {{1,2,3}}, MLP hidden dims | "
        f"K={int(best_a['sgc_k'])}, {best_a['mlp_hidden']} | "
        f"{best_a['OOT_Pooled_PRAUC_mean']:.3f} ± {best_a['OOT_Pooled_PRAUC_std']:.3f} | "
        f"{best_a['OOT_Pooled_F1_mean']:.3f} ± {best_a['OOT_Pooled_F1_std']:.3f} | "
        f"{int(best_a['n_seeds'])} |"
    )

    # Phase B: best by OOT_Pooled_PRAUC_mean
    df_b = pd.read_csv(PHASE_DIR / "sweep_phaseB" / "phaseB_aggregated.csv")
    best_b = df_b.sort_values("OOT_Pooled_PRAUC_mean", ascending=False).iloc[0]
    dir_str = "Dir=T" if best_b["use_directional_prop"] else "Dir=F"
    rows.append(
        f"| B: Graph features | Features, Direction, Topology | "
        f"{best_b['Variation']} + {dir_str} + Topo={best_b['topology']} | "
        f"{best_b['OOT_Pooled_PRAUC_mean']:.3f} ± {best_b['OOT_Pooled_PRAUC_std']:.3f} | "
        f"{best_b['OOT_Pooled_F1_mean']:.3f} ± {best_b['OOT_Pooled_F1_std']:.3f} | "
        f"{int(best_b['n_seeds'])} |"
    )

    # Phase C: best by OOT_Pooled_PRAUC_mean
    df_c = pd.read_csv(PHASE_DIR / "sweep_phaseC" / "phaseC_aggregated.csv")
    best_c = df_c.sort_values("OOT_Pooled_PRAUC_mean", ascending=False).iloc[0]
    rows.append(
        f"| C: Dropout | p ∈ {{0.1, 0.2, 0.3, 0.4}} | "
        f"p={best_c['mlp_dropout']:.1f} | "
        f"{best_c['OOT_Pooled_PRAUC_mean']:.3f} ± {best_c['OOT_Pooled_PRAUC_std']:.3f} | "
        f"{best_c['OOT_Pooled_F1_mean']:.3f} ± {best_c['OOT_Pooled_F1_std']:.3f} | "
        f"{int(best_c['n_seeds'])} |"
    )

    # Phase D: best by OOT_Pooled_PRAUC_mean
    df_d = pd.read_csv(PHASE_DIR / "sweep_phaseD" / "phaseD_aggregated.csv")
    best_d = df_d.sort_values("OOT_Pooled_PRAUC_mean", ascending=False).iloc[0]
    rows.append(
        f"| D: Optimizer | LR, Weight Decay | "
        f"LR={best_d['sgc_lr']:.4f}, WD={best_d['sgc_weight_decay']:.4f} | "
        f"{best_d['OOT_Pooled_PRAUC_mean']:.3f} ± {best_d['OOT_Pooled_PRAUC_std']:.3f} | "
        f"{best_d['OOT_Pooled_F1_mean']:.3f} ± {best_d['OOT_Pooled_F1_std']:.3f} | "
        f"{int(best_d['n_seeds'])} |"
    )

    header = (
        "| Phase | Swept | Best Config | OOT Pooled PRAUC | OOT Pooled F1 | Seeds |\n"
        "|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows)


def build_section_5() -> list:
    """Deep Res MLP — narrative only, no code cells. Table built at build time."""
    table = _build_phase_table()
    table_md = (
        "## Deep Res MLP: Greedy Phase Sweep Summary\n\n"
        "> Numbers read from `results/deep_res_mlp_results/sweep_phase*/phase*_aggregated.csv` "
        "at notebook build time. Best config per phase selected by OOT Pooled PRAUC "
        "(primary metric). All phases fixed n=3 seeds.\n\n"
        + table
        + "\n\n*Phase D slightly trails Phase C because validation PRAUC (not OOT) was used "
        "to select dropout=0.4 for the optimizer sweep; the OOT-optimal dropout was 0.3.*"
    )
    return [
        _md("---\n" + _read_narrative("deep_res_mlp_analysis.md")),
        _md(table_md),
    ]


# ── SECTION 6 ─────────────────────────────────────────────────────────────────

def build_section_6() -> list:
    """Walk-Forward Analysis: The Graph Recovery Trap."""
    cells = [_md("---\n" + _read_narrative("wf_temporal_analysis.md"))]
    cells.append(_code(
        "df_ts = pd.read_csv(f'{RESULTS_DIR}walk_forward_timesteps.csv')\n"
        "df_ts = add_parsed_columns(df_ts)\n\n"
        "# Three key models — strings validated in build pre-flight\n"
        "SGC_SWEEP  = 'Best WF: Sweep 1: SGC (baseline) (Seed 42, Var Base)'\n"
        "MLP_SWEEP  = 'Best WF: Sweep 2: + MLP Head (Seed 42, Var Base)'\n"
        "XGB_SWEEP  = 'Baseline: XGBoost WF (epsilon-fallback)'\n\n"
        "models = {\n"
        "    'SGC (baseline)': df_ts[df_ts['Sweep'] == SGC_SWEEP].drop_duplicates('Tau'),\n"
        "    'SGC+MLP':        df_ts[df_ts['Sweep'] == MLP_SWEEP].drop_duplicates('Tau'),\n"
        "    'XGBoost WF':     df_ts[df_ts['Sweep'] == XGB_SWEEP].drop_duplicates('Tau'),\n"
        "}\n"
        "palette_wf = {'SGC (baseline)': '#9b59b6', 'SGC+MLP': '#3498db', 'XGBoost WF': '#e74c3c'}\n\n"
        "fig, ax = plt.subplots(figsize=(14, 6))\n"
        "for name, sub in models.items():\n"
        "    sub = sub.sort_values('Tau')\n"
        "    ax.plot(sub['Tau'], sub['PRAUC'],\n"
        "            label=name, color=palette_wf[name], linewidth=2)\n"
        "    # Grey bands for Low-Confidence timesteps\n"
        "    for _, row in sub[sub['Low_Confidence']].iterrows():\n"
        "        ax.axvspan(row['Tau'] - 0.45, row['Tau'] + 0.45, alpha=0.15, color='grey')\n\n"
        "# Regime boundaries\n"
        "ax.axvline(42.5, color='black', linestyle='--', alpha=0.5, linewidth=1)\n"
        "ax.axvline(43.5, color='black', linestyle='--', alpha=0.5, linewidth=1)\n"
        "ymax = ax.get_ylim()[1]\n"
        "ax.text(39, ymax * 0.95, 'Pre-Shock', fontsize=9, ha='center', style='italic')\n"
        "ax.text(43, ymax * 0.95, 'Shock',     fontsize=9, ha='center', style='italic', color='red')\n"
        "ax.text(46.5, ymax * 0.95, 'Recovery', fontsize=9, ha='center', style='italic', color='#e67e22')\n\n"
        "ax.set_xlabel('Time Step τ')\n"
        "ax.set_ylabel('PRAUC  [primary metric]')\n"
        "ax.set_title(\n"
        "    'Walk-Forward PRAUC: Graph Recovery Trap vs. XGBoost Resilience\\n'\n"
        "    'Grey bands = Low-Confidence τ (N_illicit < 10)  |  n=1 seed, Seed=42'\n"
        ")\n"
        "ax.legend(loc='upper right')\n"
        "plt.tight_layout(); plt.show()"
    ))
    return cells
