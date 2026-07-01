# Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection
### The Elliptic Bitcoin Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Why does illicit-transaction detection on Bitcoin collapse at a single timestep — and is it a graph-representation failure, or something else?** We answer this with explicit falsification diagnostics rather than assumption, then engineer and measure the fixes.

---

## Table of Contents

1. [Project Overview & The Dataset](#1-project-overview--the-dataset)
2. [The τ=43 Shock: What It Is, and What It Isn't](#2-the-τ43-shock-what-it-is-and-what-it-isnt)
3. [The Graph Recovery Trap](#3-the-graph-recovery-trap)
4. [The Solutions We Engineered](#4-the-solutions-we-engineered)
5. [Experimental Results](#5-experimental-results)
6. [Repository Structure](#6-repository-structure)
7. [Installation & Usage](#7-installation--usage)
8. [Limitations](#8-limitations)

---

## 1. Project Overview & The Dataset

### The Elliptic Bitcoin Dataset

The Elliptic dataset is a **temporal transaction graph** of the Bitcoin blockchain, partitioned into **49 discrete timesteps** ($\tau = 1 \ldots 49$). Each node represents a Bitcoin transaction, and each directed edge represents the flow of funds between transactions.

| Property | Value |
|---|---|
| Total Nodes (Transactions) | 203,769 |
| Total Edges (Fund Flows) | 234,355 |
| Total Labeled Transactions | 46,564 |
| Illicit Transactions | 4,545 (9.8% of *labeled* nodes) |
| Node Features | 166 (94 local + 72 one/two-hop graph-neighborhood aggregates) |
| Temporal Span | 49 timesteps, ~2 weeks each |

Nodes carry a ternary label: **Illicit** (class 1), **Licit** (class 0), or **Unknown** (excluded from all loss and metric computation, but still an active participant in message passing and in the 72 aggregated features — see [Key Concepts](#key-concepts)). The severe class imbalance (<10% illicit pre-shock, <1% post-shock; both figures computed over *labeled* nodes) makes standard accuracy trivial and motivates **PR-AUC** and **Macro-F1** as the primary evaluation metrics, rather than raw accuracy.

### The Temporal Regime Structure

The 49 timesteps split into three experimental regimes around a single external event:

```
τ = 1–42   ┃  Pre-Shock    ┃  Illicit actors actively operating (illicit rate: 5–36% of labeled nodes)
τ = 43     ┃  Shock        ┃  AlphaBay darknet market shutdown by law enforcement (July 2017)
τ = 44–49  ┃  Recovery     ┃  Surviving actors adapt; new micro-structural patterns emerge
```

At $\tau = 43$, the count of labeled illicit nodes drops from 239 ($\tau=42$) to 24 — a **~90% collapse in a single step** — while the graph's macroscopic shape barely moves (mean degree 2.379 → 2.350; density stays flat around 0.00046). This is the project's central anomaly, and the bulk of the work below is spent adjudicating *why* every model — graph-based and tabular alike — fails at that step.

---

## 2. The τ=43 Shock: What It Is, and What It Isn't

Rather than assume a mechanism, we ran explicit falsification diagnostics — covariate-drift metrics and label-separability permutation tests — and let the committed results decide between two competing hypotheses: a collapse of the **representation** (features/topology become indistinguishable) versus a collapse of the **label distribution** (the classifier is simply starved of positive examples).

### Hypothesis A (rejected): Broadcast bias / representational collapse

**Broadcast bias** is a known GNN failure mode: when a minority-class node is surrounded by majority-class neighbors, one round of neighborhood aggregation pulls its embedding toward the majority, erasing the signal the classifier needs. If this were driving the τ=43 collapse, propagating features through the graph should make the two classes *less* separable than the raw features alone.

We tested this directly with a two-sample permutation test (10,000 permutations, 10 random seeds) between the illicit and licit feature clouds, on (a) raw node features and (b) features after one hop of symmetric graph propagation:

| Representation at τ=43 | Separable seeds (of 10) | p-value range |
|---|:---:|---|
| Raw node features | 8 / 10 | 0.003 – 0.065 |
| One-hop propagated features | **10 / 10** | 0.002 – 0.046 |

Propagation does **not** degrade separability at τ=43 — it slightly improves it. **Broadcast bias is falsified at the tested depth.** The τ=43 model failure is not porely a loss of representational information; it is a **classifier-head failure driven by class imbalance** — with only 24 illicit nodes, threshold calibration and the decision boundary become unstable even though the features remain separable. (*Caveats, stated honestly*: only single-hop propagation was tested — deeper stacks could still smear representations, so this rules out broadcast bias at K=1, not at arbitrary depth; and both separability tests at τ=43 are flagged low-confidence in the source data because n=24.)

### Hypothesis B (rejected): Anomalous geometric drift at τ=43

We measured covariate drift between **consecutive** snapshots using MMD (full feature vector) and Wasserstein distance (3-component PCA projection, fixed scaler fit on training steps only):

| Timestep (τ) | MMD (τ−1→τ) | Wasserstein (τ−1→τ) |
|---|---|---|
| 42 | 0.0034 | 0.93 |
| **43 (shock)** | 0.0128 | 1.07 |
| **44** | **0.0406** | 1.41 |
| 45 | 0.0150 | 2.51 |

Read in isolation against τ=42 (a local *minimum*), τ=43 looks like a ~4× jump. Ranked against all 48 consecutive-step transitions in the series, it isn't anomalous at all: τ=43's MMD **ranks 16th of 48**, below ordinary pre-shock transitions (τ=7: 0.028, τ=41: 0.022, τ=25: 0.022); its Wasserstein distance **ranks 44th of 48** — among the smallest single-step shifts in the entire timeline. The one genuinely anomalous transition is **τ=43→44**, whose MMD (0.0406) is the series maximum. So the feature distribution is essentially undisturbed *at* the shock, and shifts most sharply *one step after* it — the network's transactional patterns only reorganize once the surviving actors adapt.

### Verdict: Label-Prior Collapse

Both falsification tests point the same way: the τ=43 catastrophe is a **Prior Probability Shift** — a change in the class prior $p(y)$ without a corresponding collapse in $p(x \mid y)$ or in the graph's geometry. The classifier head is starved of minority-class training signal, not fed a corrupted representation.

---

## 3. The Graph Recovery Trap

Falsifying broadcast bias doesn't mean graph models are fine — it relocates the failure to a different, later phase: **recovery** ($\tau \ge 44$).

During the pre-shock phase, illicit transactions operated more or less within established darknet-market infrastructure, forming characteristic local micro-motifs. Graph models propagate over these motifs and learn to classify illicit nodes partly by the *geometry of their neighborhood*, not just their raw features. When AlphaBay was seized, surviving actors re-entered the network through different local structures — sparser neighborhoods, new routing patterns. A model trained on the pre-shock motif library encounters a different structural vocabulary in recovery, and multi-hop propagation generally makes this worse, not better, because it bakes the obsolete topology deeper into every node's representation.

Walk-forward (retrain on all steps $< \tau$, evaluate at $\tau$, report Macro-F1 = the mean of per-step illicit-F1) makes this precise:

| Model | Pre-Shock (τ=35–42) Macro-F1 | Shock (τ=43) Macro-F1 | Recovery (τ=44–49) Macro-F1 |
|---|:---:|:---:|:---:|
| XGBoost (walk-forward) | **0.895** | 0.000 | **0.393** |
| Best graph model (SGC+MLP, K=2, Dir=F, Topo=early) | 0.786 | 0.000 | 0.175 |

**At the shock itself both model families score $\sim$ 0.000** — with 24 illicit nodes, neither family recovers the minority class. This is the trap to avoid on slides: *tabular models do not "survive" τ=43*; they collapse identically. Their real edge is **faster recovery** (0.393 vs 0.175 Macro-F1 over τ=44–49) and a higher pre-shock ceiling — not shock-robustness.

**A tempting mechanism, correctly labeled as unproven:** of the 166 features, 72 are graph-neighborhood aggregates and 94 are purely local, so a tree could in principle stop splitting on the aggregates once the neighborhood signal becomes stale, while a GNN blends local and neighborhood information at every layer and cannot opt out. This is a plausible story for the *recovery* gap — but the committed experiments include **no feature-importance analysis and no 94-local-only vs. 166-all-features ablation** to confirm it. We report it as a hypothesis for future work, not a measured result — see [Limitations](#8-limitations).

---

## 4. The Solutions We Engineered

### Exponential Temporal Decay — mitigating structural overfitting

The pathology behind the Graph Recovery Trap is **stale geometry in the training window**: a walk-forward model trained on all history equally weights pre-shock motifs that no longer represent current illicit behavior. We introduce exponential decay on the per-sample loss weight, $w_t = \lambda^{\tau - t}$, so more recent topology dominates training.

| Configuration | Baseline WF Macro-F1 | + Decay WF Macro-F1 | Best λ |
|---|:---:|:---:|:---:|
| XGBoost (decayed multiscale graph features) | 0.634 | **0.674** | 0.50 |
| SGC K=2, Dir=T, Topo=late | 0.437 | **0.527** | 0.25 |
| SGC K=2, Dir=T, Topo=None | 0.442 | 0.501 | 0.25 |
| SGC K=3, Dir=F, Topo=late | 0.472 | 0.490 | 0.50 |

Decay pushes the overall benchmark ceiling to **0.674 Macro-F1** (XGBoost + decayed multiscale graph features, λ=0.50) and crowns a new best *pure-graph* configuration at **0.527 Macro-F1** (directed SGC K=2 with late topology injection, λ=0.25) — surpassing the previous non-decayed graph champion (0.489). Fast decay (λ=0.50) suits XGBoost, which greedily splits on the freshest available signal; medium decay (λ=0.25) suits the directed, late-topology SGC variant, which otherwise overfits to exact directed-path structure. At τ=43 itself, decay keeps both families weakly active (XGBoost+decay: 0.154 vs. 0.000 undecayed; decay-champion SGC: 0.118 vs. 0.000), and both recover materially faster through τ=44–49.

### PCA as a Geometric Regularizer for Deep Neighborhoods

As SGC propagation depth $K$ increases, each node's representation incorporates an exponentially larger neighborhood — a form of oversmoothing where node-level specificity is progressively lost. We measured intrinsic dimensionality (a proxy for information content, via two-NN estimation) of propagated embeddings:

| Setting | K=1 Mean Intrinsic Dim | K=2 Mean Intrinsic Dim | K=3 Mean Intrinsic Dim |
|---|---|---|---|
| Base (no PCA) | 7.68 | 7.96 | 7.57 |
| PCA (99% variance) | 7.33 | 7.32 | **6.82** |

Without PCA, going from K=2 to K=3 already shows early oversmoothing (7.96 → 7.57). PCA fitted on the training split *before* propagation amplifies this compression at K=3 (down to 6.82), acting as a controlled regularizer that removes redundant dimensions introduced by deep multi-hop mixing before they overfit to pre-shock structure. Because the PCA basis is fixed on training steps only, this cannot leak information from future timesteps.

### Classifier-Head Ablation — smaller and non-residual wins

We swept the SGC classifier head architecture over four phases (depth/width → graph feature controls → dropout → optimizer), reporting **static out-of-time (OOT) Macro PR-AUC** on τ=35–49 as primary metric (this window includes the full shock+recovery stress test, unlike the pre-shutdown development split). The initial hypothesis — that a *deep residual* MLP would help — was **not supported**: residual variants underperformed a compact, non-residual head at every scale tested. The winning configuration:

- **SGC parameters:** K=3, PCA features (98% variance), directional message passing, late topology injection
- **Head:** 2 hidden layers `(64, 64)`, LayerNorm, SiLU activations, **no residual connections**, dropout 0.3
- **Optimizer:** AdamW, LR = 0.01, WD = 5e-4

**Final static OOT performance (τ=35–49):**

| Metric | Value |
|---|---|
| OOT Macro PR-AUC | **0.305** |
| OOT Macro F1 | **0.262** |

Wider heads and residual connections consistently made things worse: extra capacity made it easier for the head to memorize pre-shock graph motifs, which is exactly the failure mode the rest of this project is about. This tuned graph head still does **not** beat the strongest tabular baselines (RandomForest, XGBoost) on OOT Macro metrics — its value is in showing how far a purely graph-based pipeline can be pushed before hitting that ceiling.

---

## 5. Experimental Results

### Walk-Forward Regime Breakdown (Macro-F1; retrain on all τ' < τ, evaluate at τ)

| Configuration | Pre-Shock (35–42) | Shock (43) | Recovery (44–49) | WF Macro-F1 |
|---|:---:|:---:|:---:|:---:|
| SGC (K=2, Dir=F, Topo=None), linear head | 0.480 | 0.016 | 0.130 | 0.309 |
| SGC + MLP (K=3, Dir=F, Topo=None), no multiscale | 0.549 | 0.000 | 0.235 | 0.387 |
| SGC + MLP + MP (K=3, Dir=T, Topo=late) | 0.687 | 0.000 | 0.189 | 0.442 |
| SGC + MLP + MP (K=2, Dir=F, Topo=early) — best graph, no decay | 0.786 | 0.000 | 0.175 | **0.489** |
| SGC + decay (K=2, Dir=T, Topo=late, λ=0.25) — best pure-graph overall | — | 0.118 | — | **0.527** |
| XGBoost (walk-forward) | 0.895 | 0.000 | 0.393 | 0.634 |
| XGBoost + decay (λ=0.50) — best overall | — | 0.154 | — | **0.674** |

Static OOT baselines (train τ≤26, evaluate τ=35–49, no retraining): LogReg 0.241 Macro-F1, PyG GCN 0.208 (170.1 s), RandomForest 0.479 (6.7 s), XGBoost 0.475 (2.9 s) — XGBoost trains ~59× faster than the GCN baseline while matching or beating it. Walk-forward retraining lifts every model, but the *ranking* is unchanged: tabular > graph, throughout.

---

## 6. Repository Structure

```
.
├── source/
│   ├── config.py                    # Config dataclass, DEVICE, OUTPUT_DIR, set_global_seeds()
│   ├── sweep.py                     # Main entry point: runs the full ablation grid
│   ├── build_eda_data.py            # Generates EDA CSVs (drift, homophily, PCA, t-SNE, PageRank)
│   ├── extract_eda_stats.py         # Per-step EDA statistic extraction
│   ├── ensemble_ablation.py         # Ensemble experiment runner
│   ├── reproduce_weber.py           # Reproduction of Weber et al.'s original LR/RF baselines
│   ├── data/
│   │   ├── load_dataset.py          # KaggleHub loader; validates temporal edge integrity
│   │   ├── build_graph.py           # EllipticDataModule: per-snapshot graph construction,
│   │   │                            #   SGC propagation, scaler fitting, PCA preprocessing
│   │   ├── temporal_features.py     # Lag-windowed snapshot-level features (leakage-guarded)
│   │   └── snapshot_topology.py     # Raw per-snapshot graph statistics
│   ├── models/
│   │   ├── layers.py                # gcn_norm, sgc_propagate (symmetric, multiscale, directional)
│   │   ├── classifier.py            # SGCHead (linear or MLP head via MLPBlock)
│   │   └── temporal_head.py         # SnapshotEmbedder, TemporalLSTM, SnapshotEMA (superseded — see §8)
│   ├── evaluation/
│   │   ├── validation.py            # fit_head, stack_prop, walk_forward_validation,
│   │   │                            #   threshold calibration with ε-fallback
│   │   ├── temporal_validation.py   # LSTM/EMA conditioned walk-forward loops (superseded — see §8)
│   │   ├── ablation_validation.py   # XGBoost tabular walk-forward; per-τ CSV logging
│   │   ├── wf_metrics.py            # stratified_wf_metrics: regime-stratified F1/PR-AUC
│   │   └── falsification_log.py     # World-A/B/C/γ diagnostic logging
│   ├── analysis/
│   │   ├── tda_diagnostic.py        # Topological Data Analysis; MMD canary at τ=43
│   │   ├── label_separability.py    # Permutation MMD tests for representational collapse
│   │   ├── temporal_analysis.py     # Per-step temporal plots and seed aggregation
│   │   ├── grid_intrinsic_dim.py    # Intrinsic dimensionality estimation across the grid
│   │   └── check_topology_leak.py   # Audit for topology-level temporal leakage
│   ├── experiments/
│   │   └── local_only_prop_experiment.py  # 94-local-vs-166-all ablation harness (unrun — see §8)
│   └── reporting/
│       ├── build_presentation_notebook.py # Regenerates presentation/*.ipynb from results/
│       ├── check_sweep_names.py     # Validates sweep name consistency across CSVs
│       ├── check_thesis.py          # Automated thesis-gate reporting
│       └── results/                 # Markdown analysis reports (this README summarizes these)
├── presentation/
│   └── elliptic_bitcoin_math_presentation.ipynb  # Generated presentation notebook + figures
├── results/
│   ├── sweep_results.csv            # CSV-1: per-variation aggregate metrics (Base rows)
│   ├── final_aggregated_results.csv # Per-variation metrics incl. PCA rows (Variation=PCA)
│   ├── walk_forward_timesteps.csv   # CSV-2: per-τ step F1/PR-AUC/Precision/Recall
│   ├── final_aggregated_timesteps.csv # Per-τ metrics incl. decay configurations
│   ├── snapshot_topology.csv        # Raw graph statistics per timestep
│   ├── label_separability.csv       # Permutation separability test results (raw vs. propagated)
│   ├── eda_drift.csv                # MMD and Wasserstein drift per timestep (consecutive-step)
│   ├── falsification_log.csv        # Broadcast-bias / drift verdict ledger
│   ├── topological_diagnostics.csv  # TDA canary results
│   ├── deep_res_mlp_results/        # Phase A–D classifier-head sweep outputs
│   └── figures/                     # Generated plots and diagnostic figures
├── tests/
│   ├── test_remediation.py          # Adversarial suite: leakage and integrity guards
│   ├── test_eda_drift.py            # Unit tests for MMD and Wasserstein estimators
│   ├── test_ablation_validation.py
│   ├── test_sweep_xgb_baseline.py
│   ├── test_sweep_parser.py
│   └── test_presentation_sweep_names.py
├── REPORTS_ACCURACY_AUDIT.md        # Referee-style audit of every claim in reporting/results/
├── REPORTS_CORRECTED_PASSAGES.md    # Drop-in corrected passages applied per the audit
└── references/                      # Reference papers
```

---

## 7. Installation & Usage

### Prerequisites

- Python 3.10+
- The Elliptic dataset (fetched automatically via KaggleHub on first run — requires a Kaggle account and `~/.kaggle/kaggle.json`, cached at `~/.cache/kagglehub/datasets/ellipticco/elliptic-data-set/versions/1`)

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

This runs the full ablation grid across SGC configurations, MLP head variations, XGBoost baselines, and temporal-decay ablations. Results are written to `results/sweep_results.csv` (per-variation aggregates) and `results/walk_forward_timesteps.csv` (per-τ breakdowns).

### Running Individual Analyses

```bash
# Snapshot topology statistics
python -m source.data.snapshot_topology

# Label separability permutation tests (broadcast-bias falsification)
python source/analysis/label_separability.py

# Temporal regime analysis and plots
python source/analysis/temporal_analysis.py

# EDA drift, homophily, PageRank, PCA/t-SNE metrics
python source/build_eda_data.py

# Regenerate the presentation notebook from the current results/ CSVs
python -m source.reporting.build_presentation_notebook
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

All tests are self-contained (no network required) and use synthetic data and mocks. `test_remediation.py` enforces temporal leakage guards across feature scaling, edge validation, threshold calibration, and walk-forward block construction.

---

## 8. Limitations

Stated explicitly so the project's claims aren't over-read:

- **The tabular-recovery mechanism is unproven.** The "trees discard stale graph-aggregate features" explanation for why XGBoost recovers faster than SGC is a hypothesis; no feature-importance analysis or 94-local-vs-166-all ablation was run to confirm it (harness exists but is unrun at `source/experiments/local_only_prop_experiment.py`).
- **Broadcast bias is ruled out only at one hop.** The separability test that falsifies it uses K=1 propagation; deeper multi-layer stacks were not tested for representational smearing.
- **The shock step is statistically thin.** Only ~24 labeled illicit nodes exist at τ=43; separability tests and Macro-F1 there are flagged low-confidence and should not be over-interpreted as a single reliable number.
- **Limited seeds, no significance testing.** Most grid configurations ran 2–3 seeds (some single-seed); small Macro-F1 gaps (~0.01) should not be treated as reliable rankings.
- **Evaluation ignores the unlabeled majority.** ~75% of nodes are `unknown` and excluded from loss/metrics, yet they participate in neighborhood features and message passing; reported performance characterizes only the labeled subgraph.
- **Discrete-snapshot scope.** All models operate on 49 discrete snapshots. Continuous-time approaches (TGN, EvolveGCN) were considered but not implemented — see `source/reporting/results/conclusion.md` for the full future-work discussion.
- **Drift metrics are unconditional.** Covariate drift is measured over all nodes, not class-conditional on illicit status, so it evidences whole-network reorganization rather than illicit-specific migration.
- **LSTM/EMA temporal-memory heads** (`source/models/temporal_head.py`, `source/evaluation/temporal_validation.py`) are retained in the codebase but superseded by the SGC-ablation framing described here; an earlier project phase falsified a narrower "broadcast graph-level fusion" variant of them (see `results/falsification_log.csv`), which does not generalize to a claim that temporal memory is never useful — see `conclusion.md` for scoped future work (per-node temporal state, attention over histories).

---

## Key Concepts

| Term | Definition |
|---|---|
| **Walk-Forward (WF)** | At each test step τ, retrain from scratch on all prior steps and evaluate at τ. The primary, deployment-style evaluation protocol; Macro-F1 = mean of per-step illicit-F1. |
| **PR-AUC** | Threshold-free average precision on the illicit class; primary metric for static comparisons. |
| **Prior Probability Shift** | A change in the marginal class distribution $P(y)$ without a corresponding change in $P(x|y)$. What happens at τ=43, confirmed by falsifying both broadcast bias and drift-onset alternatives. |
| **Broadcast Bias** | Neighborhood aggregation homogenizing minority-class embeddings toward the majority. Falsified at τ=43 (K=1 propagation improves separability, not degrades it). |
| **Graph Recovery Trap** | A graph model achieves near SOTA pre-shock performance while remaining structurally tethered to obsolete micro-motifs, causing it to underperform in the post-shock recovery regime relative to tabular models. |
| **Temporal Decay** | Exponential sample re-weighting $w_t = \lambda^{\tau-t}$ applied during WF training to down-weight stale historical topology. |
| **ε-Fallback** | When the threshold-calibration step has fewer than ε=10 illicit nodes, calibration falls back to a local-quantile estimator to avoid noise-driven miscalibration. |
| **Intrinsic Dimensionality** | A proxy for information diversity in node embeddings, estimated via two-NN methods; collapses under oversmoothing. |
