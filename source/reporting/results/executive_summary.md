## Abstract

We study illicit-transaction detection on the **Elliptic Bitcoin dataset** — a sequence of 49 time-ordered, directed transaction-graph snapshots — under severe temporal non-stationarity, centered on the $\tau=43$ shock corresponding to the July 2017 law-enforcement takedown of the AlphaBay darknet market. Across exploratory data analysis, graph-falsification diagnostics, a large Simple Graph Convolution (SGC) architecture sweep, tabular and deep-graph baselines, walk-forward evaluation, and exponential-decay feature weighting, we reach three headline conclusions. **(1)** The catastrophic $\tau=43$ performance collapse is *not* a geometric or representational collapse: node features — raw and graph-propagated — remain class-separable at the shock; the failure is a **label-prior collapse** (labelled illicit nodes drop ~90% in one step) that destabilizes the classifier head. **(2)** **Tabular tree ensembles** (XGBoost, RandomForest) dominate out-of-time evaluation and recover faster after the shock than any graph model we tested. **(3)** The strongest overall configuration is **XGBoost augmented with exponentially decaying multiscale graph features** (0.674 walk-forward Macro-F1), while the best *pure-graph* model is a directed SGC with late topology injection and decay (0.527). Full code, the data-processing pipeline, and all result tables are available at **[github.com/AgricolaC/Elliptic-Bitcoin-Project](https://github.com/AgricolaC/Elliptic-Bitcoin-Project)**.

## Introduction

Detecting illicit activity in cryptocurrency transaction graphs is a high-stakes anomaly-detection problem: the positive (illicit) class is a small, shifting minority, adversaries adapt, and the underlying network evolves over time. The Elliptic dataset is the canonical public benchmark for this task, providing 166 features per transaction (94 local, 72 graph-neighborhood aggregates) across 49 sequential time-step snapshots, with nodes labelled illicit, licit, or unknown.

The dataset's defining challenge is **temporal non-stationarity**, epitomized by the $\tau=43$ snapshot: the AlphaBay shutdown removes the dominant illicit subpopulation almost overnight, and essentially every model — graph-based and tabular alike — fails catastrophically at that step. This project asks *why* that failure happens and *which modeling choices are robust to it*. We deliberately separate two hypotheses that are often conflated — a collapse of the feature/topology **representation** versus a collapse of the **label distribution** — and adjudicate between them with explicit falsification diagnostics rather than assuming either.

Our study spans per-snapshot EDA (degree, PageRank, homophily, PCA/t-SNE embeddings); falsification diagnostics (covariate-drift metrics and label-separability permutation tests); an SGC architecture grid (neighborhood depth, directionality, multiscale propagation, PCA compression, topology injection); tabular and deep-graph baselines; walk-forward (deployment-style) evaluation; and exponential-decay weighting of graph features. The complete implementation and every result artifact behind the numbers in this report are openly available at **[github.com/AgricolaC/Elliptic-Bitcoin-Project](https://github.com/AgricolaC/Elliptic-Bitcoin-Project)** — readers are encouraged to consult the repository for the exact pipeline, hyperparameters, and CSV outputs.

### Key Findings

Our analysis reveals five core findings regarding model performance and the nature of the network shock:

1. **The Graph Structure:** The Elliptic graph is a sequence of directed transaction snapshots $G_\tau=(V_\tau,E_\tau)$. The macroscopic structure of the network (volume, density, mean degree) remains stable throughout the entire timeline.
2. **The Nature of the Shock ($\tau=43$):** The catastrophic failure of machine learning models post-shock likely is not caused by a sudden geometric collapse of the feature space. Instead, it is driven by a massive **class-prior collapse**.
3. **Graph Feature Learning:** Techniques like Simple Graph Convolution (SGC) propagation and PCA uncover useful graph structures and homophily before the shock. However, these graph features are highly sensitive to specific micro-motifs.
4. **Topological Overfitting (a recovery-phase effect):** Deep graph models overfit to the local motifs of the pre-shock illicit economy. The single largest step-to-step covariate shift occurs at the $\tau=43\rightarrow44$ transition (the series-maximum MMD), marking where the post-shock network re-organizes; the graph models fail to generalize into recovery because they remain tethered to the now-obsolete connectivity patterns of the pre-shock era. (This MMD is measured over *all* nodes, so it evidences whole-network re-organization, not an illicit-specific migration per se.)
5. **The Tabular Advantage (and what does *not* explain it):** Tabular models (XGBoost, RandomForest) are the strongest benchmark on out-of-time Macro-F1. A tempting explanation is feature selection: of the 166 dataset features, 72 are 1-hop/2-hop graph aggregates and 94 are purely local, so a decision tree could stop splitting on the aggregates when the neighborhood signal degrades, whereas a GCN blends local and neighborhood features at every layer and cannot opt out. **We flag this as a hypothesis, not a measured result.** Two facts constrain it: (i) at the shock itself ($\tau=43$) XGBoost also scores $0.000$ Macro-F1 — it does *not* survive the shock, so its edge is not shock-robustness; and (ii) the committed experiments contain no feature-importance breakdown or 94-vs-166 ablation to confirm that trees discard the aggregates. The defensible, verified claim is empirical: tabular models **recover faster** after the shock (Recovery Macro-F1 $0.393$ vs $\sim0.175$ for the best graph model) and hold a higher pre-shock ceiling. 

## Data Model and Notation

At each time step $\tau \in \{1,\dots,49\}$, we define a directed graph snapshot:

$$
G_\tau=(V_\tau,E_\tau,X_\tau,y_\tau),
$$

where each node $v \in V_\tau$ represents a Bitcoin transaction, and each directed edge $e \in E_\tau$ represents a flow of funds from one transaction to another. 

Each node possesses a feature vector $X_\tau$ (local and aggregated metrics) and a label:

$$
y_i \in \{0,1,-1\}
$$

where labels correspond to **licit ($0$)**, **illicit ($1$)**, or **unknown ($-1$)**. Unknown labels are excluded from the loss calculation and evaluation metrics.

> **The Epistemological Gap of the `Unknown` Class**: While `Unknown` nodes are masked during loss calculation, they are computationally active participants in the feature space. The 72 aggregated tabular features are pre-computed over neighborhoods that *include* these unlabelled nodes, and standard GNNs pass messages through them. Because `Unknown` simply means a lack of forensic evidence (they could be laundering intermediaries or benign retail users), the neighborhood features and message-passing arrays blindly ingest massive amounts of epistemological noise. Robust models must explicitly utilize feature-selection mechanisms (like decision trees) or attention-based pooling to selectively ignore this unlabelled noise when it corrupts the neighborhood signal.

The supervised learning task is highly imbalanced: the positive (illicit) class represents a very small minority of the global distribution (typically $<10\%$ pre-shock, and $<1\%$ post-shock). Because the negative class vastly outnumbers the positive class, **OOT Macro PR-AUC** (for static comparisons) and **Walk-Forward Macro Illicit-F1** (for regime tracking) are the primary metrics for evaluating model performance.
