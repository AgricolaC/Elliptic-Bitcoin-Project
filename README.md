# Elliptic Bitcoin Anomaly Detection: Temporal Graph Analysis

This repository contains an advanced Graph Neural Network (GNN) and Temporal Analysis framework designed for the **Elliptic Bitcoin Dataset**. The project specifically targets the challenges of semi-supervised anomaly detection (identifying illicit transactions) in highly non-stationary, adversarial financial networks.

## 📌 Project Overview

Unlike traditional static graph datasets, the Elliptic dataset is characterized by extreme temporal drift and an abrupt "regime shift" at timestep 43 (the shutdown of a major Darknet market). 

This project explores why traditional GNNs and linear dimensionality reduction techniques (like PCA) fail spectacularly during these structural shocks. We implement a rigorous **Walk-Forward** validation architecture, and temporal modeling via LSTMs and Exponential Decay.

### Key Contributions
- **Phased Walk-Forward Pipeline:** A strict out-of-time evaluation framework that completely prevents temporal data leakage.
- **Manifold Drift Diagnostics:** An implementation of Incremental PCA to mathematically prove the geometric breakdown of feature variance during regime shifts.
- **Topological Invariants & Directionality:** Support for both undirectional/directional message passing and topological feature injection (e.g., PageRank).
- **Automated Reporting Suite:** A pipeline that compiles execution metrics directly into a master Jupyter Notebook presentation.

---

## 🏗 System Architecture

The codebase is structured to enforce modularity and prevent memory/temporal hazards:

```text
Elliptic Bitcoin Project/
│
├── source/
│   ├── analysis/             # Analytical diagnostics (Intrinsic Dimension, Temporal Drift, Topology Leaks)
│   ├── data/                 # Graph construction, temporal batching, and IPCA pipelines
│   ├── evaluation/           # Walk-Forward temporal validation logic
│   ├── execution/
│   │   ├── phases/           # Core execution phases (F1: Walk-Forward, F2: LSTM, F3: Baselines, F4: Decay)
│   │   └── run_pipeline.py   # Entry point for the experimental pipeline
│   ├── models/               # PyTorch Geometric neural network definitions (SGC, GCN, etc.)
│   ├── reporting/            # Automated generation of presentation.ipynb and thesis checks
│   ├── config.py             # Global hyperparameter configurations
│   └── sweep.py              # Static Grid Search evaluation
│
├── results/                  # Generated CSV metrics and evaluation logs
├── presentation.ipynb        # Master presentation notebook (auto-generated)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PyTorch & PyTorch Geometric (PyG)
- CUDA (Highly Recommended for large matrix multiplications)
- XGBoost, scikit-learn, NetworkX, Pandas

### Installation
1. Clone the repository and navigate to the project root.
2. Install the required dependencies:
   ```bash
   pip install -r source/requirements.txt
   ```

### Dataset
Download version 1 of the `ellipticco/elliptic-data-set` dataset through KaggleHub before running the pipeline. The loader expects KaggleHub's cached files under:

```text
~/.cache/kagglehub/datasets/ellipticco/elliptic-data-set/versions/1
```

Ensure you have approximately 2GB of free disk space.

---

## ⚙️ Execution Pipeline

The core experiments are divided into four distinct phases. The runner executes one required phase per invocation:

```bash
python source/execution/run_pipeline.py --phase f1
python source/execution/run_pipeline.py --phase f2
python source/execution/run_pipeline.py --phase f3
python source/execution/run_pipeline.py --phase f4
```

Run all four commands to execute the complete phased pipeline.

### Phase Details
* **Phase F1 (Walk-Forward):** Executes selected configurations from the static grid search through a strict, one-step-ahead Walk-Forward temporal validation. Includes the Incremental PCA tracking to diagnose manifold shifts. The checked-in winner list currently contains the directional K=3 PCA configuration; the base XGBoost and K=2 SGC calls remain available but commented out.
* **Phase F2 (LSTM):** Evaluates Temporal Graph Networks by injecting historical node embeddings into an LSTM backbone.
* **Phase F3 (Baselines):** Computes Reference Baselines including raw XGBoost, Multi-Layer Perceptron (MLP), and foundational 2-layer Graph Convolutional Networks (GCN).
* **Phase F4 (Exponential Decay):** Applies exponential temporal decay to historical training samples to mitigate the impact of stale pre-shock graph structures. The checked-in runner currently executes the SGC decay grid; XGBoost decay is temporarily disabled to avoid duplicating existing CSV rows.

---

## 📊 Analytical Diagnostics

The `source/analysis/` directory contains standalone scripts for diagnosing mathematical behaviors of the graph:

* **Intrinsic Dimension (`grid_intrinsic_dim.py`):** Calculates the intrinsic fractal dimension (using MLE) of the node embeddings across different propagation depths ($K$) and directionality settings.
* **Temporal Drift (`temporal_analysis.py`):** Plots the degradation of model performance exactly at the $\tau=43$ regime shift.
* **Topology Leakage (`check_topology_leak.py`):** A strict security check to ensure no edges accidentally cross the strict temporal boundaries.

## 📝 Automated Reporting

Once you have executed the pipeline and the results CSV files are populated in the `results/` directory, you can automatically generate the final presentation:

```bash
python source/reporting/build_presentation.py
```
This script will parse the metrics, generate the relevant matplotlib plots, and dynamically construct `presentation.ipynb`.

---
