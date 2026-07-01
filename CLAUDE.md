# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project applying Simple Graph Convolution (SGC) with optional MLP/XGBoost/LSTM heads to illicit Bitcoin transaction detection on the Elliptic dataset. The key finding: τ=43 (AlphaBay darknet shutdown) causes **label-prevalence collapse** (90% drop in illicit nodes), not representational collapse — node embeddings remain separable at τ=43.

## Setup

```bash
source venv/bin/activate
```

The Elliptic dataset is fetched automatically from KaggleHub. Requires the cache to exist at `~/.cache/kagglehub/datasets/ellipticco/elliptic-data-set/versions/1`. The download runs on first call to `download_and_load_data()`.

## Running Tests

```bash
# All tests (from repo root)
python -m pytest tests/

# Single test class or method
python -m pytest tests/test_remediation.py::TestFeatureScaling::test_topology_columns_are_scaled_W1

# Fast no-network tests only (all current tests are unit/mock-based)
python -m pytest tests/ -x
```

Test files require no network; they use synthetic data and mocks. The adversarial suite in `test_remediation.py` guards against temporal leakage regressions (W1–W8 labels reference the original remediation workbook).

## Running Analyses

```bash
# Main ablation sweep (writes to results/sweep_results.csv + walk_forward_timesteps.csv)
python source/sweep.py

# Snapshot topology stats
python -m source.data.snapshot_topology

# Per-step EDA and temporal analysis
python source/analysis/temporal_analysis.py
python source/extract_eda_stats.py
```

Analysis modules under `source/analysis/` each have a `__main__` block. Run them from the repo root with the venv active.

## Architecture

```
source/
  config.py                  — Config dataclass (all hyperparams), DEVICE, OUTPUT_DIR, set_global_seeds()
  data/
    load_dataset.py          — download_and_load_data(); temporal edge validation (_validate_temporal_edges)
    build_graph.py           — EllipticDataModule: per-snapshot graph building, SGC propagation, feature scaling
    temporal_features.py     — Lag-windowed snapshot-level temporal features (leakage-guarded)
    snapshot_topology.py     — Raw per-snapshot graph statistics (node/edge counts, illicit rate, degree)
  models/
    layers.py                — gcn_norm, sgc_propagate (symmetric-normalized, multiscale, optional directional)
    classifier.py            — SGCHead (Linear or deep residual MLP via MLPBlock), build_loss
    temporal_head.py         — SnapshotEmbedder, TemporalLSTM, SnapshotEMA, LSTMConditionedHead
  evaluation/
    validation.py            — fit_head, stack_prop, walk_forward_validation, threshold calibration
    temporal_validation.py   — LSTM/EMA walk-forward conditioned training loops
    ablation_validation.py   — XGBoost tabular walk-forward, CSV-2 per-τ logging
    wf_metrics.py            — stratified_wf_metrics: regime-stratified F1/PRAUC aggregation
    falsification_log.py     — World-A/B/C/γ diagnostic logging
  sweep.py                   — Ablation sweep runner (main entry point); writes both CSVs
  analysis/                  — Standalone diagnostic modules (TDA, label separability, etc.)
  reporting/                 — Result summarization scripts and markdown analysis reports
results/                     — Output CSVs and figures (gitignored: results/models/, archive/)
tests/                       — Pytest suite (adversarial leakage guards + unit tests)
```

## Key Concepts

**Temporal split** (from `Config`): train τ∈[1,26], val τ∈[27,34], test τ∈[35,49]. Disruption step τ=43.

**Three regimes** (from `wf_metrics.py`): `pre_shock` (τ≤42), `shock` (τ=43), `recovery` (τ≥44). All aggregate metrics are reported regime-stratified.

**Primary metric**: PRAUC (average precision) — threshold-free. F1 is secondary; it requires threshold calibration (see `_calibrate_threshold` in `validation.py`). At τ=43 the calibration step can have <10 illicit nodes, triggering local-quantile fallback.

**Label encoding**: illicit=1, licit=0, unknown=-1. Only labeled nodes (label∈{0,1}) enter loss and metrics.

**SGC propagation**: `sgc_propagate` in `layers.py` returns `[X | SX | ... | S^K X]` (multiscale) or `S^K X` alone. `S = D^{-1/2}(max(A,Aᵀ)+I)D^{-1/2}`. Optional directional channels add row-normalized in/out-degree propagations.

**Leakage guards**: temporal edge validation (`_validate_temporal_edges`) catches cross-step and orphan edges before any graph is built. Timestep `ts` is explicitly excluded from feature columns (`_select_feature_cols`). Feature scaler is fitted on training steps only.

**Walk-forward validation (WF)**: at each test step τ, the model is trained on all steps <τ and evaluated at τ. This is the primary evaluation protocol (not the static train/val/test split).

**CSV outputs**:
- `results/sweep_results.csv` (CSV-1): one row per sweep variation × seed; pooled + macro + regime metrics.
- `results/walk_forward_timesteps.csv` (CSV-2): one row per τ × sweep; per-step F1/PRAUC/Precision/Recall.

## Import Convention

All `source/` modules import as if `source/` is on `sys.path` (e.g., `from config import Config`, `from data.build_graph import EllipticDataModule`). Test files explicitly add both `REPO_ROOT` and `SOURCE_DIR` to `sys.path` at the top.
