# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin transaction anomaly detection using Scalable Graph Convolution (SGC) on the Elliptic dataset. The project implements temporal walk-forward validation with systematic ablation sweeps to evaluate different architectural components.

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run standard ablation sweep (seed 42)
python source/sweep.py --mode standard

# Run mega sweep (seeds 42-44, multiple variations)
python source/sweep.py --mode mega

# Run all tests
pytest tests/test_remediation.py -v

# Run specific test class
pytest tests/test_remediation.py::TestFeatureScaling -v
```

Results are saved to `results/sweep_results.csv`. Models are saved to `results/models/`.

## Architecture

### Data Pipeline (`source/data/`)

```
Elliptic Dataset (203K nodes, 49 timesteps)
    ↓ load_dataset.py (downloads from Kaggle, validates temporal integrity)
    ↓ build_graph.py: EllipticDataModule
        - Per-timestep graph construction
        - Base scaler: StandardScaler on 166 raw features (train steps only)
        - Optional topology: PageRank + Clustering Coefficient
        - Augmented scaler: Second StandardScaler on [base|topo]
        - SGC propagation: S^k where S = D^{-1/2}(A+A^T+I)D^{-1/2}
        - Output: [X | SX | ... | S^K X] multiscale features
```

### Model (`source/models/`)

- `layers.py`: SGC propagation with symmetric normalization, supports directional channels (in/out/undirected)
- `classifier.py`: SGCHead - 3-layer MLP (input → 128 → 64 → 2) or linear classifier

### Evaluation (`source/evaluation/`)

- `fit_head`: Trains classifier on specified timesteps
- `walk_forward_validation`: Expanding window validation - train on [1..tau-1], test on tau for each tau ∈ [35..49]

### Configuration (`source/config.py`)

Key settings in `EllipticConfig` dataclass:
- `train_steps=range(1,35)`, `test_steps=range(35,50)`, `disruption_step=43`
- Architecture toggles: `use_mlp_head`, `use_multiscale_prop`, `use_graph_structural`, `use_directional_prop`
- SGC hyperparameters: `sgc_k=2`, `sgc_epochs=200`, `sgc_lr=0.01`, `sgc_weight_decay=5e-4`

## Ablation Sweep Design

Each sweep toggles exactly one flag relative to the previous:
1. **Sweep 1 (Baseline):** Linear SGC head
2. **Sweep 2 (+ MLP):** 3-layer MLP head
3. **Sweep 3 (+ Multiscale):** [X|SX|S²X] stacking
4. **Sweep 4 (+ Structure):** PageRank + clustering coefficients
5. **Sweep 5 (+ Directional):** In/Out/Undirected propagation channels

## Test Suite

`tests/test_remediation.py` uses "axiom-falsify" pattern - tests should FAIL before fixes exist:
- W1: Feature scaling for topology features
- W3: Temporal leakage guards (no cross-timestep edges)
- W5: Result dict 11-key schema standardization
- W6: sgc_input_dim encapsulation
- W7: Plot filename collision prevention
- W8: Dynamic class weights per-tau

## Critical Domain Knowledge

**Structural Hysteresis (Step 43):** Dark market shutdown causes graph topology shift. Models overfit to pre-disruption structure, causing performance degradation. Sliding windows (4-step memory) recover F1 by "amputating toxic geometry."

**XGBoost Baseline:** Tabular attributes are more elastic than graph structure - XGBoost achieves 0.871 WF F1 vs SGC's 0.625 because raw features survive the topology shift.

## Dependencies

PyTorch, scikit-learn, pandas, numpy, NetworkX, XGBoost, matplotlib, joblib, pytest
