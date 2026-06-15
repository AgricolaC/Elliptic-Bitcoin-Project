# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin transaction anomaly detection using Scalable Graph Convolution (SGC) on the Elliptic dataset (203K nodes, 49 timesteps). Implements temporal walk-forward validation with systematic ablation sweeps to evaluate architectural components.

## Commands

```bash
# Activate virtual environment (required before any python command)
source venv/bin/activate

# Run standard ablation sweep (seed 42 only, Base variation)
python source/sweep.py --mode standard

# Run mega sweep (seeds 42-44, Base/PCA/RF_Pruned variations)
python source/sweep.py --mode mega

# Run only static OOT evaluation (skip walk-forward, faster)
python source/sweep.py --only-static

# Run only walk-forward evaluation (skip static OOT)
python source/sweep.py --only-wf

# Run all tests
pytest tests/test_remediation.py -v

# Run a single test class
pytest tests/test_remediation.py::TestFeatureScaling -v

# Run a single test method
pytest tests/test_remediation.py::TestFeatureScaling::test_topology_columns_are_scaled_W1 -v
```

Results are saved to `results/sweep_results.csv`. Per-step walk-forward data goes to `results/walk_forward_timesteps.csv`. Models are saved to `results/models/`.

## Architecture

### Import Path

`source/` is the Python root. All imports within source use unqualified names (`from config import Config`, `from data.build_graph import ...`). Tests add both the repo root and `source/` to `sys.path`.

### Data Pipeline (`source/data/`)

```
download_and_load_data()          # load_dataset.py — Kaggle cache at ~/.cache/kagglehub/
    ↓ _validate_temporal_edges()  # W3 guard: raises on orphan or cross-timestep edges
    ↓ EllipticDataModule.setup()  # build_graph.py
        1. build per-timestep graphs (reindex_timestep: global txIds → contiguous [0..n-1])
        2. scaler_base: StandardScaler on 166 raw features (train steps only)
        3. optional topology: PageRank + Clustering Coefficient per node via NetworkX
           - 'early' mode: append to x_np before second scaler pass
           - 'late' mode: store separately in g["topo"], concatenated after SGC propagation
        4. scaler_aug/scaler_topo: second StandardScaler pass (W1 fix — topology was unscaled)
        5. SGC propagation via sgc_propagate() → stored in g["prop"]
        6. optional PCA or RF feature selection on propagated features
        → sets dm.sgc_input_dim from actual tensor shape (W6 fix)
```

Feature dimensions after propagation:
- Undirected multiscale K=2: `166 * (K+1) = 498`
- Directional multiscale K=2: `166 * (1 + 3*K) = 830`
- Late topology injection adds +2 to any of the above

### SGC Propagation (`source/models/layers.py`)

- `gcn_norm`: symmetric D^{-1/2}(A + A^T + I)D^{-1/2} — symmetrizes the DAG
- `_row_normalize`: D^{-1}A for directional channels
- `sgc_propagate`: returns `[X | SX | ... | S^K X]` (multiscale) or `S^K X` (standard). With `use_directional=True`, returns `[X | S_sym X | S_out X | S_in X | ...]` per hop.

### Model (`source/models/classifier.py`)

`SGCHead`: either a single Linear (baseline) or a configurable MLP. MLP architecture is driven by `cfg.mlp_hidden` (tuple of hidden dims), `cfg.mlp_dropout`, and `cfg.use_residual`. Residual shortcut is `nn.Identity` if in_dim == last hidden dim, else `nn.Linear`.

### Evaluation Flow (`source/evaluation/validation.py`)

Two evaluation modes, both using `fit_head` → `SGCHead` trained with AdamW + CrossEntropyLoss:

1. **Static OOT**: Train on all `train_steps` [1..26], validate on `val_steps` [27..34], test on all `test_steps` [35..49] pooled. (P0-A fix: 3-way split).
2. **Walk-forward**: For each tau in [35..49], train on [start..tau-2], calibrate threshold on tau-1, test on tau. Class weights recomputed per-tau from the actual training window (W8 fix). Produces `walk_forward_drift_{sweep_name}.png`.

`_aggregate_walk_forward` returns both pooled (concatenated) and macro-averaged F1/PR-AUC.

**Frozen Preprocessing in Walk-Forward.** Scalers, topology, PCA, and propagation matrices are fit once on `train_steps` and held constant across the walk-forward loop. Only the classification head is retrained per-τ. This is a stated simplification, not leakage.

**Validation Split Limitation.** Validation (steps 27–34) sits entirely on the pre-disruption side of the step-43 regime change. Selection is therefore made on pre-disruption data, and we expect — and observe — a generalization gap at τ≥43. This is inherent to the dataset: the post-disruption regime can only exist in the test window.

### Sweep Runner (`source/sweep.py`)

Three execution phases:
1. **Phase 1**: Sweep 1 (linear SGC) → Sweep 2 (+ MLP head) as static-only baselines
2. **Phase 2**: Grid search over `K ∈ {1,2,3}`, `directional ∈ {F,T}`, `topo ∈ {F,T}`, `injection ∈ {early,late}` — all static-only. Grid skips `(topo=False, injection='early')` as duplicate.
3. **Phase 2.5**: MLP head variations (Wide/Residual/ResWide) on champion + challenger configs
4. **Phase 3**: Walk-forward on best static-OOT SGC config only

Sweep results are checkpointed to CSV after each sweep — re-runs skip already-completed sweep names. Artifact names sanitized with `re.sub(r"[^\w\-]", "_", name)` (W7 fix).

`_make_result()` enforces the 11-key schema on every result dict (W5 fix).

### Configuration (`source/config.py`)

`Config` dataclass with `__post_init__` assertion that `train_steps` and `test_steps` are disjoint. `DEVICE` auto-selects CUDA → MPS → CPU. `set_global_seeds()` seeds Python/NumPy/PyTorch and sets cuDNN deterministic mode.

## Test Suite

`tests/test_remediation.py` follows the **axiom-falsify** pattern — each test was written to FAIL before the corresponding fix, then verified to PASS after. Do not remove assertions even if they seem redundant.

| Class | What it guards |
|---|---|
| `TestFeatureScaling` | W1: topology columns (PageRank/clustering) must be StandardScaled; leakage guard that scaler is fitted on train only |
| `TestTemporalLeakageGuard` | W3: orphan edge txIds must raise (old NaN==NaN bypass bug) |
| `TestSGCInputDimEncapsulation` | W6: `dm.sgc_input_dim` set inside `setup()`, not by callers |
| `TestWalkForwardPlotNaming` | W7: `walk_forward_validation` must have `sweep_name` parameter |
| `TestDynamicClassWeights` | W8: `walk_forward_validation` must NOT accept external `cls_w` |
| `TestSweepResultKeyStandardization` | W5: all result dicts must have the exact 11 canonical keys |

## Critical Domain Knowledge

**Label encoding**: Class `"1"` = illicit (positive, `y=1`), class `"2"` = licit (`y=0`), `"unknown"` = unlabeled (`y=-1`). Unlabeled nodes are excluded from loss and metric computation everywhere via `labeled_mask` / `y != -1`.

**Structural Hysteresis (Step 43)**: A dark market shutdown causes a topology shift at timestep 43. Graph-structural models overfit pre-disruption topology and degrade here. Sliding window walk-forward (4-step memory) partially recovers by "amputating toxic geometry."

**XGBoost dominates SGC**: Raw tabular features are more topology-shift-elastic. XGBoost achieves ~0.871 walk-forward F1 vs SGC's ~0.625 because tabular features survive the step-43 disruption.

**Directional propagation requires multiscale**: `sgc_propagate` asserts `multiscale=True` when `use_directional=True`. Setting `use_directional_prop=True` with `use_multiscale_prop=False` will raise at runtime.

## Temporal Limitations and Related Work

### LSTM Extrapolation Limitation
The LSTM is trained on snapshots 1–26. At evaluation, it must produce hidden states h_τ for τ up to 49, including the step-43 regime change it never saw during training. The hidden-state dynamics in the post-disruption regime are untrained extrapolation. This is inherent to any temporal model trained on pre-disruption data and is not fixable without leaking test information into training.

### Computational Cost
The temporal module itself (LSTM forward over ~26 vectors) is cheap. However, walk-forward evaluation retrains the entire embedder+LSTM+head pipeline end-to-end from scratch for each of 15 test steps at 100 epochs each, with per-epoch loss summed over all labeled training nodes. This repeated retraining dominates runtime cost.

### EvolveGCN
Related work such as EvolveGCN models the evolution of the graph by placing an RNN over the GCN weight matrices. In contrast, our SGC-LSTM approach places the LSTM over the compressed graph-level embeddings and concatenates the output with the node-level propagated features. Our approach uses static graph convolutions (SGC) without learnable parameters during the message passing phase, which is vastly cheaper than evolving GCN weights, while still capturing macroscopic temporal shifts over the ~26 training snapshots.
