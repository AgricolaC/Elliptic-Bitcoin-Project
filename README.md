# Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection
### The Elliptic Bitcoin Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Detecting illicit cryptocurrency transactions in a highly non-stationary temporal graph, with a focus on robustness across structural regime shifts.**

---

## Table of Contents

1. [Project Overview & The Dataset](#1-project-overview--the-dataset)
2. [Core Discoveries & The Graph Recovery Trap](#2-core-discoveries--the-graph-recovery-trap)
3. [The Solutions We Engineered](#3-the-solutions-we-engineered)
4. [Experimental Results](#4-experimental-results)
5. [Repository Structure](#5-repository-structure)
6. [Installation & Usage](#6-installation--usage)

---

## 1. Project Overview & The Dataset

### The Elliptic Bitcoin Dataset

The Elliptic dataset is a **temporal transaction graph** of the Bitcoin blockchain, partitioned into **49 discrete timesteps** ($\tau = 1 \ldots 49$). Each node represents a Bitcoin transaction, and each directed edge represents the flow of funds between transactions.

| Property | Value |
|---|---|
| Total Nodes (Transactions) | 203,769 |
| Total Edges (Fund Flows) | 234,355 |
| Total Labeled Transactions | 46,564 |
| Illicit Transactions | 4,545 (9.8% of labeled) |
| Node Features | 166 (local + aggregated graph features) |
| Temporal Span | 49 timesteps, ~2 weeks each |

Nodes carry a ternary label: **Illicit** (class 1), **Licit** (class 0), or **Unknown** (excluded from all loss and metric computation). The severe class imbalance (~10% illicit) makes standard accuracy trivial and motivates the use of **PR-AUC** as the primary, threshold-free evaluation metric.

### The Temporal Regime Structure

The 49 timesteps are divided into three experimental regimes, separated by a singular macroeconomic event:

```
τ = 1–42   ┃  Pre-Shock    ┃  Illicit actors actively operating (illicit rate: 5–32%)
τ = 43     ┃  Shock        ┃  AlphaBay darknet market shutdown by law enforcement
τ = 44–49  ┃  Recovery     ┃  Surviving actors adapt; new micro-structural patterns emerge
```

### The Challenge: A Prior Probability Shift at $\tau = 43$

On July 4, 2017, the FBI and Europol seized the AlphaBay darknet marketplace — one of the largest illicit Bitcoin exchanges in history. In the dataset, this manifests as a sudden and catastrophic **Prior Probability Shift** at $\tau = 43$.

The macroscopic network geometry **did not change**. The graph remained structurally similar to the pre-shock phase:

| Timestep | N_illicit | Illicit Rate | Mean Degree | Regime |
|---|---|---|---|---|
| $\tau = 42$ | **239** | 11.1% | 2.379 | Pre-Shock |
| $\tau = 43$ | **24** | 1.75% | 2.350 | Shock |
| $\tau = 44$ | **24** | 1.51% | 2.232 | Recovery |
| $\tau = 45$ | **5** | 0.41% | 2.384 | Recovery |

The mean degree at $\tau = 43$ (2.350) is indistinguishable from the pre-shock average (2.259 ± 0.164). Yet the **volume of illicit transactions dropped by ~90% overnight** — from 239 labeled illicit nodes at $\tau = 42$ to just 24 at $\tau = 43$. This is a **Prior Probability Shift** (a change in the class prior), not a geometric or representational shift.

We rigorously confirmed this via **permutation-based label separability tests** across 10 random seeds at $\tau = 43$: raw features remain statistically separable (p < 0.05) in **8 out of 10** trials. The embeddings are not broken. The classifier head is simply starved of minority-class training signal.

---

## 2. Core Discoveries & The Graph Recovery Trap

### The Falsification of Representational Collapse

A widespread hypothesis in the literature frames the $\tau = 43$ degradation as *representational collapse* or *broadcast bias* — the idea that graph propagation homogenizes the node embeddings until illicit and licit nodes become indistinguishable. **We falsified this hypothesis.**

The covariate drift metrics (Maximum Mean Discrepancy and Wasserstein distance on PCA projections) tell a precise story:

| $\tau$ | MMD (Feature Drift) | Wasserstein (PCA Drift) |
|---|---|---|
| 42 | 0.0034 | 0.93 |
| **43** | **0.0128** | **1.07** |
| **44** | **0.0406** | **1.40** |
| 45 | 0.0150 | 2.51 |

$\tau = 43$ exhibits only *moderate* geometric drift. The **actual** large-scale structural reorganization of the feature manifold occurs at **$\tau = 44$** — one step *after* the shock — as the surviving actors adapt their transaction patterns. There is no geometric correlate for model failure at $\tau = 43$ itself.

> **Key Finding:** $\tau = 43$ is a **label-deprivation event**. The classifier head collapses to predicting the majority class because it has almost no positive examples on which to calibrate — not because the graph embeddings have been corrupted.

### Topological Overfitting and The Graph Recovery Trap

The regime shift at $\tau \ge 44$ reveals a deeper and more consequential failure mode: **Topological Overfitting**.

During the pre-shock phase ($\tau \le 42$), illicit transactions operated within established darknet market infrastructure, forming characteristic local micro-motifs (e.g., fan-out patterns from market addresses, concentrated exchange routing). Graph-based models, by propagating information across these motifs, learn to classify illicit nodes by the *geometry of their neighborhood* — not merely by their raw features.

When law enforcement dismantled AlphaBay, the surviving illicit actors were forced to reorganize entirely. They re-entered the network through **different local structures** — sparser neighborhoods, different hub-and-spoke patterns, novel transaction routing strategies. A GNN trained on the pre-shock micro-motif library encounters a fundamentally different structural vocabulary in the recovery phase.

The results are stark. Across all SGC configurations tested without temporal adaptation, the **pooled recovery F1 never exceeds 0.26**, even when pre-shock F1 exceeds 0.82:

| Model | WF Pre-43 F1 | WF Shock F1 | WF Recovery F1 |
|---|---|---|---|
| SGC K=1 (baseline) | 0.535 | 0.016 | 0.095 |
| SGC K=2, Dir=F (best pooled) | 0.822 | 0.000 | 0.259 |
| SGC K=3, Dir=T + early Topo (PCA) | 0.767 | 0.056 | 0.360 |

This is the **Graph Recovery Trap**: a graph model can achieve state-of-the-art performance on the pre-shock regime while simultaneously being entirely blind to the post-shock recovery, because the topology it learned no longer exists.

### Tabular Resilience

Tabular baselines that operate on **localized node features** — without graph propagation — are structurally immune to Topological Overfitting. XGBoost in a Walk-Forward setup achieves:

| Model | WF Pre-43 F1 | WF Shock F1 | WF Recovery F1 | WF Pooled F1 |
|---|---|---|---|---|
| XGBoost (WF, ε-fallback) | **0.902** | 0.000 | **0.472** | **0.834** |
| Best SGC (K=2, Dir=F) | 0.822 | 0.000 | 0.259 | 0.713 |
| SGC + MLP Head | 0.731 | 0.013 | 0.105 | 0.530 |

XGBoost's recovery F1 (0.472) is **1.82× higher** than the best vanilla SGC configuration (0.259). Because XGBoost predicts from each node's 166 raw features independently, it is insensitive to changes in the graph's mesoscopic structure. When the network reorganizes post-shock, the per-node feature distributions shift — but at a much slower pace — and the tabular model adapts gracefully.

---

## 3. The Solutions We Engineered

### Temporal Decay — The Cure for Topological Overfitting

The fundamental pathology of Topological Overfitting is **stale geometry in the training set**. A model trained continuously using all historical data implicitly assigns equal weight to pre-shock topological patterns that no longer represent current illicit behavior. The cure is to make the model *actively forget* obsolete micro-motifs.

In a **Walk-Forward (WF) training protocol**, at each test step $\tau$, the model is re-trained on all preceding steps $[1, \tau-1]$. We introduce an **exponential time decay** on the per-sample loss contribution:

$$w_t = \lambda^{\tau - t}, \quad \lambda \in (0, 1)$$

A higher $\lambda$ (slower decay) preserves more historical information; a lower $\lambda$ forces the model to weight recent topology more aggressively. The effect on recovery performance is dramatic:

| Model + Decay | WF Shock F1 | WF Recovery F1 | WF Pooled F1 |
|---|---|---|---|
| XGBoost (no decay) | 0.000 | 0.472 | 0.834 |
| XGBoost + λ=0.05 | 0.000 | 0.541 | 0.846 |
| XGBoost + λ=0.25 | 0.154 | 0.488 | 0.831 |
| **XGBoost + λ=0.5** | **0.154** | **0.604** | 0.836 |

With $\lambda = 0.5$, recovery F1 improves from **0.472 to 0.604** — a **+28% relative gain** — confirming that temporal down-weighting of historical topology is an effective and principled mitigation.

Applied to SGC-based models, temporal decay on the training window similarly improves recovery, with the best SGC+decay configurations achieving recovery F1 values near 0.45, closing more than half of the gap to XGBoost.

### PCA as a Geometric Regularizer for Deep Neighborhoods

As the SGC propagation depth $K$ increases, each node's representation incorporates information from an exponentially larger neighborhood. At $K = 3$ in particular, node features become heavily influenced by the global network structure — a form of **mathematical oversmoothing** in which node-level specificity is progressively lost.

We measured the **intrinsic dimensionality** (a proxy for information content in the representation) of propagated embeddings across grid configurations:

| Setting | $K=1$ Mean Intrinsic Dim | $K=2$ Mean Intrinsic Dim | $K=3$ Mean Intrinsic Dim |
|---|---|---|---|
| Base (no PCA) | 7.68 | 7.96 | 7.57 |
| PCA (99% variance) | 7.33 | 7.32 | **6.82** |

Without PCA, going from $K=2$ to $K=3$ reduces intrinsic dimensionality from 7.96 to 7.57, signalling the onset of oversmoothing. **PCA applied before propagation amplifies this compression** — at $K=3$, PCA reduces intrinsic dimensionality to 6.82, acting as a **controlled regularizer** that removes redundant dimensions introduced by deep multi-hop mixing before they can overfit to the pre-shock graph structure.

This is not merely a dimensionality reduction trick: PCA defines a fixed basis fitted on the training split, ensuring the dimensionality-reduction step cannot leak information from future timesteps.

### Deep Residual MLP Head

Through a sequential four-phase hyperparameter sweep (architecture depth → graph features → dropout → optimizer), we found that the classifier head architecture has an outsized impact on out-of-time generalization.

The final **Deep Residual MLP** configuration:

- **SGC Parameters:** $K = 3$, PCA features, directional message passing, late topological feature injection  
- **MLP Architecture:** 2 hidden layers `(64, 64)`, LayerNorm, SiLU activations  
- **Regularization:** Dropout $p = 0.3$, AdamW (LR $= 0.01$, WD $= 10^{-4}$)

Notably, smaller hidden layers `(64, 64)` significantly outperformed wider architectures `(128, 128)` and `(256, 128)`. A large classifier head with access to rich propagated features can memorize the pre-shock geometry far more efficiently than a narrow bottleneck — wider is strictly worse in the recovery regime.

**Peak out-of-time performance:**

| Metric | Value |
|---|---|
| OOT Pooled Illicit-F1 | **0.4827** |
| OOT Macro F1 | **0.2622** |

---

## 4. Experimental Results

### Walk-Forward Regime Breakdown (All Major Configurations)

| Configuration | Pre-43 F1 | Shock F1 | Recovery F1 | Pooled WF F1 | WF PR-AUC |
|---|---|---|---|---|---|
| SGC Baseline (K=1) | 0.535 | 0.016 | 0.095 | 0.338 | 0.307 |
| SGC + MLP Head | 0.731 | 0.013 | 0.105 | 0.530 | 0.624 |
| SGC K=2, Dir=F, Topo=None | 0.822 | 0.000 | 0.259 | **0.713** | 0.703 |
| SGC K=3, Dir=F, Topo=late (PCA) | 0.787 | 0.000 | 0.234 | 0.696 | **0.739** |
| SGC K=3, Dir=T, Topo=early (PCA) | 0.767 | 0.056 | **0.360** | 0.679 | 0.670 |
| **XGBoost WF (ε-fallback)** | **0.902** | 0.000 | 0.472 | 0.834 | 0.890 |
| XGBoost + Decay λ=0.5 | 0.884 | **0.154** | **0.604** | 0.836 | 0.870 |

All Walk-Forward results use a threshold calibration protocol with an epsilon-fallback for the $\tau = 43$ shock step, where the calibration split may contain fewer than 10 illicit nodes.

---

## 5. Repository Structure

```
.
├── source/
│   ├── config.py                 # Config dataclass, device, seeds, output paths
│   ├── sweep.py                  # Main entry point: runs the full ablation grid
│   ├── build_eda_data.py         # Generates EDA CSVs (drift, homophily, PCA, t-SNE)
│   ├── ensemble_ablation.py      # Ensemble experiment runner
│   ├── data/
│   │   ├── load_dataset.py       # KaggleHub loader; validates temporal edge integrity
│   │   ├── build_graph.py        # EllipticDataModule: per-snapshot graph construction,
│   │   │                         #   SGC propagation, scaler fitting, PCA preprocessing
│   │   ├── temporal_features.py  # Lag-windowed snapshot-level features (leakage-guarded)
│   │   └── snapshot_topology.py  # Raw per-snapshot graph statistics
│   ├── models/
│   │   ├── layers.py             # gcn_norm, sgc_propagate (symmetric, multiscale, directional)
│   │   ├── classifier.py         # SGCHead (linear or deep residual MLP via MLPBlock)
│   │   └── temporal_head.py      # SnapshotEmbedder, TemporalLSTM, SnapshotEMA,
│   │                             #   LSTMConditionedHead
│   ├── evaluation/
│   │   ├── validation.py         # fit_head, stack_prop, walk_forward_validation,
│   │   │                         #   threshold calibration with ε-fallback
│   │   ├── temporal_validation.py# LSTM/EMA conditioned walk-forward training loops
│   │   ├── ablation_validation.py# XGBoost tabular walk-forward; CSV-2 per-τ logging
│   │   ├── wf_metrics.py         # stratified_wf_metrics: regime-stratified F1/PR-AUC
│   │   └── falsification_log.py  # World-A/B/C/γ diagnostic logging
│   ├── analysis/
│   │   ├── tda_diagnostic.py     # Topological Data Analysis; Betti number diagnostics
│   │   ├── label_separability.py # Permutation MMD tests for representational collapse
│   │   ├── temporal_analysis.py  # Per-step temporal plots and seed aggregation
│   │   ├── grid_intrinsic_dim.py # Intrinsic dimensionality estimation across grid
│   │   └── check_topology_leak.py# Audit for topology-level temporal leakage
│   └── reporting/
│       ├── check_sweep_names.py  # Validates sweep name consistency across CSVs
│       ├── check_thesis.py       # Automated thesis-gate reporting
│       └── results/              # Markdown analysis reports (EDA, TDA, ablation)
├── results/
│   ├── sweep_results.csv         # CSV-1: per-variation aggregate metrics
│   ├── walk_forward_timesteps.csv# CSV-2: per-τ step F1/PR-AUC/Precision/Recall
│   ├── snapshot_topology.csv     # Raw graph statistics per timestep
│   ├── label_separability.csv    # Permutation separability test results
│   ├── eda_drift.csv             # MMD and Wasserstein drift per timestep
│   ├── deep_res_mlp_results/     # Phase A–D deep residual MLP sweep outputs
│   └── figures/                  # Generated plots and diagnostic figures
├── tests/
│   ├── test_remediation.py       # Adversarial suite: 50+ leakage and integrity guards
│   ├── test_eda_drift.py         # Unit tests for MMD and Wasserstein estimators
│   ├── test_ablation_validation.py
│   └── test_sweep_xgb_baseline.py
└── references/                   # Reference papers
```

---

## 6. Installation & Usage

### Prerequisites

- Python 3.10+
- The Elliptic dataset (fetched automatically via KaggleHub on first run — requires a Kaggle account and `~/.kaggle/kaggle.json`)

### Setup

```bash
git clone <repository-url>
cd "Elliptic Bitcoin Project"

python -m venv venv
source venv/bin/activate

pip install -r source/requirements.txt
```

### Running the Main Ablation Sweep

```bash
# From the repository root, with the venv active:
python source/sweep.py
```

This runs the full ablation grid across all SGC configurations, MLP variations, XGBoost baselines, and temporal decay ablations. Results are written to `results/sweep_results.csv` (per-variation aggregates) and `results/walk_forward_timesteps.csv` (per-$\tau$ breakdowns).

### Running Individual Analyses

```bash
# Snapshot topology statistics
python -m source.data.snapshot_topology

# Label separability permutation tests
python source/analysis/label_separability.py

# Temporal regime analysis and plots
python source/analysis/temporal_analysis.py

# EDA drift metrics (MMD, Wasserstein, homophily)
python source/build_eda_data.py
```

### Running the Test Suite

```bash
# Full test suite
python -m pytest tests/

# Single adversarial test class
python -m pytest tests/test_remediation.py::TestTemporalLeakageGuard -v

# Single test
python -m pytest tests/test_remediation.py::TestFeatureScaling::test_topology_columns_are_scaled_W1
```

All tests are self-contained (no network required) and use synthetic data and mocks. The adversarial suite in `test_remediation.py` enforces strict temporal leakage guards across feature scaling, edge validation, threshold calibration, and walk-forward block construction.

---

## Key Concepts

| Term | Definition |
|---|---|
| **Walk-Forward (WF)** | At each test step τ, retrain from scratch on all prior steps. The only evaluation protocol that respects temporal ordering. |
| **PR-AUC** | Primary metric. Threshold-free average precision on the illicit class. |
| **Prior Probability Shift** | A change in the marginal class distribution p(y) without a corresponding change in p(x\|y). What happens at τ=43. |
| **Topological Overfitting** | A model learns to classify based on neighborhood micro-motifs that are specific to a historical network structure, not transferable to post-shock reorganizations. |
| **Temporal Decay** | Exponential sample re-weighting w_t = λ^(τ−t) applied during WF training to down-weight stale historical topology. |
| **ε-Fallback** | When the calibration step has fewer than ε=10 illicit nodes, threshold selection falls back to a local-quantile estimator to avoid noise-driven miscalibration. |
| **Intrinsic Dimensionality** | A proxy for information diversity in node embeddings, estimated via two-NN methods. Collapses under oversmoothing. |
