## Diagnostic & Falsification Analysis: The $\tau=43$ Anomaly

> **Why we did this**: To rigorously test our hypothesis that the $\tau=43$ anomaly was caused by a sudden collapse in geometric feature space or graph topology (Broadcast Bias). We needed to know if the features broke, or if it was simply a label-prior collapse.

Our intuition of this dataset was that the catastrophic model failure at $\tau=43$ was due to "Representational Collapse" or "Graph Structure Drift". We performed this multi-faceted diagnostic analysis to rigorously test and falsify our hypothesis.

A critical challenge in the Elliptic Bitcoin dataset is the notorious performance degradation around time step $\tau=43$ (widely associated with the darknet market shutdown of AlphaBay). Our initial hypothesis has been that this event caused **"Representational Collapse"** or **"Broadcast Bias"**—the idea that the underlying geometric features or graph structure suddenly shifted, making illicit and licit nodes indistinguishable.

By triangulating `falsification_log.csv`, `label_separability.csv`, `topological_diagnostics.csv`, and `eda_drift.csv`, we can falsify this hypothesis.

### 1. Covariate Drift (`eda_drift.csv`): The Onset of Geometric Shift


| Time Step ($\tau$) | MMD (Feature Drift) | Wasserstein (PCA Drift) |
| :---: | :---: | :---: |
| 42 | 0.0034 | 0.93 |
| **43** | **0.0128** | **1.07** |
| **44** | **0.0406** | **1.40** |
| 45 | 0.0150 | 2.50 |
| 46 | 0.0294 | 2.79 |

* **Evaluating the Jump**: At $\tau=43$, MMD jumps from $0.0034$ to $0.0128$—a nearly **4x increase** in feature drift exactly at the timestep of the shock. While $\tau=44$ sees an even larger compounding effect, the geometric drift absolutely begins at $\tau=43$.

### 2. Label Separability (`label_separability.csv`): Raw Features Survive, but Broadcast Bias Remains Unfalsified

Our separability tests demonstrate that at $\tau=43$, across 10 random seeds, the **raw features** are distinctly separable (`p < 0.05`) in 8 out of 10 trials. 

* **The Limitation**: This proves that the *raw nodal features* did not collapse. However, it fails to falsify **Broadcast Bias** in Graph Neural Networks. Broadcast Bias is a topological phenomenon occurring *during message passing*. If the remaining illicit nodes are structurally isolated and surrounded by licit neighbors (high heterophily), a GCN or GraphSAGE will aggregate those dominant licit features, hopelessly smearing the hidden representations. To truly rule out Broadcast Bias, we must test the separability of the **graph-convolved embeddings** ($L$-th layer hidden state), not just the raw inputs.

### 3. The True Culprit: Subpopulation Shift vs. Pure Prior Shift

The catastrophic drop in illicit nodes is undeniable:

* $\tau=42$: **239** illicit nodes
* **$\tau=43$: 24 illicit nodes** (a ~90% catastrophic drop)
* $\tau=44$: 24 illicit nodes

While initially framed as a pure **Prior Probability Shift** (assuming $P(Y)$ changes while $P(X|Y)$ remains constant), this assumes the 24 surviving nodes are a random, identically distributed subset of the original 239. 

Given the darknet shutdown targeted a specific, dominant market (AlphaBay), this is highly unlikely. The surviving 24 nodes likely represent entirely different classes of illicit behavior (e.g., isolated ransomware, small-time tumblers) occupying a fundamentally different region of the feature space and possessing a different subgraph topology. Therefore, $\tau=43$ is not just a Prior Shift; it is a violent **Subpopulation Shift**. The models fail because they are starved of minority examples, and the remaining examples are geometric and topological aliens compared to the dominant signature trained on in $\tau \le 42$.

### Conclusion & Modeling Takeaways

The reality of the $\tau=43$ anomaly is complex and multi-faceted. Our revised understanding suggests a simultaneous, interacting set of shifts:

1. **Subpopulation Shift at $\tau=43$**: The sudden death of the dominant illicit subpopulation causes a massive prior shift but also instantly alters the topological positioning of the minority class.
2. **Onset of Geometric Drift ($\tau=43 \rightarrow 44$)**: Feature drift begins immediately (4x MMD jump at $\tau=43$) and compounds violently at $\tau=44$.
3. **Vulnerability to Broadcast Bias**: While raw features remain separable at $\tau=43$, standard message-passing is likely highly destructive for the remaining topological outliers.

The massive jump in MMD drift at $\tau=44$ serves as the formal diagnostic signature of the network’s structural adaptation following the shock. While the initial drop at $\tau=43$ was a Prior Shift, the compounding geometric divergence at $\tau=44$ confirms that the illicit subpopulation has fundamentally migrated to new transaction patterns. Because deep graph models were trained on the pre-shock illicit economy, their reliance on historical local motifs creates a form of 'Topological Overfitting'; they are effectively pattern-matching against an illicit landscape that no longer exists. Consequently, when these models encounter the recovery-phase actors—who now occupy distinct regions of the feature manifold—they fail to generalize, treating these new nodes as background noise rather than actionable illicit anomalies.

**Actionable Modeling Strategy**: Relying solely on loss-reweighting (for Prior Shift) is insufficient if the underlying embeddings are being smeared by Broadcast Bias. We must:
- 1. **Test graph-convolved embeddings** to quantify the true extent of Broadcast Bias.
- 2. Implement **heterophily-aware GNN architectures** or **ego-centric sampling** to protect isolated illicit nodes from being overwhelmed by licit neighbors during message passing.
- 3. Employ techniques robust to **subpopulation shifts**, rather than assuming the remaining illicit nodes match the historical distribution.
