## Diagnostic & Falsification Analysis: The $\tau=43$ Anomaly

> **Why we did this**: To rigorously test our hypothesis that the $\tau=43$ anomaly was caused by a sudden collapse in geometric feature space or graph topology (Broadcast Bias). We needed to know if the features broke, or if it was simply a label-prior collapse.

Our intuition of this dataset was that the catastrophic model failure at $\tau=43$ was due to "Representational Collapse" or "Graph Structure Drift". We performed this multi-faceted diagnostic analysis to rigorously test and falsify our hypothesis.

A critical challenge in the Elliptic Bitcoin dataset is the notorious performance degradation around time step $\tau=43$ (widely associated with the darknet market shutdown of AlphaBay). Our initial hypothesis has been that this event caused **"Representational Collapse"** or **"Broadcast Bias"**—the idea that the underlying geometric features or graph structure suddenly shifted, making illicit and licit nodes indistinguishable.

By triangulating `falsification_log.csv`, `label_separability.csv`, `topological_diagnostics.csv`, and `eda_drift.csv`, we can falsify this hypothesis.

### 1. Covariate Drift (`eda_drift.csv`): No Anomalous Feature Shift *at* the Shock

We measure covariate drift as the distributional distance between **consecutive** snapshots ($\tau-1 \rightarrow \tau$), on features standardized with a scaler fit only on the training steps. Two metrics are used: **MMD** (Maximum Mean Discrepancy, an RBF-kernel two-sample statistic) on the full feature vector, and the **Wasserstein / optimal-transport distance** on a fixed 3-component PCA projection.

| Time Step ($\tau$) | MMD ($\tau-1\rightarrow\tau$) | Wasserstein ($\tau-1\rightarrow\tau$) |
| :---: | :---: | :---: |
| 42 | 0.0034 | 0.93 |
| **43 (shock)** | 0.0128 | 1.07 |
| **44** | **0.0406** | 1.41 |
| 45 | 0.0150 | 2.51 |
| 46 | 0.0294 | 2.79 |

* **The $\tau=43$ "jump" is a baseline artifact**: Against $\tau=42$ alone, the step to $\tau=43$ looks like a ~4x MMD increase—but $\tau=42$ is a local *minimum* of the drift series. Ranked against all 48 consecutive-step transitions, the $\tau=43$ MMD ($0.0128$) ranks only **16th of 48**, below ordinary pre-shock transitions such as $\tau=7$ ($0.028$), $\tau=41$ ($0.022$), and $\tau=25$ ($0.022$); and the $\tau=43$ Wasserstein ($1.07$) ranks **44th of 48—among the smallest single-step shifts in the entire timeline**. There is no anomalous covariate drift *at* the shock.
* **The real anomaly is one step later**: The $\tau=43\rightarrow44$ transition carries the **largest MMD of the whole series** ($0.0406$). The feature distribution is essentially undisturbed at the moment illicit nodes vanish ($\tau=43$) and shifts most sharply only *afterward* ($\tau=44$). This is the signature of a **label-prior (prevalence) shift**, not a feature-space collapse, and it is consistent with the project's overall finding that covariate drift is stable at the shock itself.

### 2. Label Separability (`label_separability.csv`): The Failure Is at the Classifier Head, Not the Representation

**Broadcast Bias** is a known failure mode of message-passing GNNs: when a minority-class node is surrounded by majority-class neighbors, one round of neighborhood aggregation pulls its embedding toward the majority, erasing the signal. If the $\tau=43$ collapse were caused by Broadcast Bias, propagating features through the graph would make the classes *less* separable than the raw node features. We test this directly, running a two-sample permutation test between the illicit and licit feature clouds (separable = permutation $p<0.05$) at $\tau=43$ across 10 seeds, on (a) **raw** node features and (b) features after **one hop of symmetric graph propagation** (the aggregation a GNN layer performs):

| Representation | Separable seeds (of 10) | $p$-value range |
| --- | :---: | --- |
| Raw node features | 8 / 10 | 0.003 – 0.065 |
| One-hop propagated features | **10 / 10** | 0.002 – 0.046 |

* **Broadcast Bias is falsified at the tested depth**: Propagation does *not* degrade separability at $\tau=43$—it slightly improves it (8→10 seeds). The illicit class is still linearly distinguishable *after* the aggregation that Broadcast Bias would have destroyed. The catastrophic $\tau=43$ failure is therefore **not** a loss of representational information; it is a **classifier-head failure driven by class imbalance**—with only 24 illicit nodes, threshold calibration and the decision boundary become unstable even though the features remain separable.
* **Two honest caveats**: (i) only single-hop propagation was tested, so Broadcast Bias is ruled out *at $K=1$*, not at arbitrary depth; (ii) the $\tau=43$ test rests on only 24 illicit nodes and is flagged low-confidence—the direction (separable, consistent across all 10 seeds) is trustworthy, but the effect size at this single step should not be over-interpreted.

### 3. The True Culprit: Subpopulation Shift vs. Pure Prior Shift

The catastrophic drop in illicit nodes is undeniable:

* $\tau=42$: **239** illicit nodes
* **$\tau=43$: 24 illicit nodes** (a ~90% catastrophic drop)
* $\tau=44$: 24 illicit nodes

While initially framed as a pure **Prior Probability Shift** (assuming $P(Y)$ changes while $P(X|Y)$ remains constant), this assumes the 24 surviving nodes are a random, identically distributed subset of the original 239. 

Given the darknet shutdown targeted a specific, dominant market (AlphaBay), this is highly unlikely. The surviving 24 nodes likely represent entirely different classes of illicit behavior (e.g., isolated ransomware, small-time tumblers) occupying a fundamentally different region of the feature space and possessing a different subgraph topology. Therefore, $\tau=43$ is not just a Prior Shift; it is a violent **Subpopulation Shift**. The models fail because they are starved of minority examples, and the remaining examples are geometric and topological aliens compared to the dominant signature trained on in $\tau \le 42$.

### Conclusion & Modeling Takeaways

Triangulating the four diagnostics, the $\tau=43$ anomaly resolves into a clear picture:

1. **Label-prior (prevalence) collapse at $\tau=43$**: The dominant illicit subpopulation vanishes (239→24 labelled illicit nodes), an order-of-magnitude drop in the positive base rate. As argued in Section 3, the ~24 survivors are also a *non-random* subsample (subpopulation shift), so they need not match the historical illicit distribution.
2. **No geometric shift *at* the shock**: Covariate drift at $\tau=43$ is unremarkable (16th of 48 steps by MMD; near-lowest by Wasserstein). The largest single-step feature shift is the $\tau=43\rightarrow44$ transition (series-maximum MMD)—the network re-organizes *after* the shock, not during it.
3. **The representation survives; the classifier head does not**: Both raw and one-hop-propagated features remain class-separable at $\tau=43$, so Broadcast Bias is falsified at the tested depth. The $\tau=43$ failure is head-level class imbalance, not representational collapse.

That post-shock re-organization ($\tau\ge44$) is where deep graph models struggle: having baked the pre-shock neighborhood topology into their features, they generalize poorly once the recovery-phase graph diverges—a form of *topological overfitting* against an illicit landscape that no longer exists. This is a recovery-phase effect, distinct from the $\tau=43$ imbalance failure above.

**Actionable Modeling Strategy**: Because the $\tau=43$ failure is head-level imbalance (not smeared embeddings), the levers are:
- 1. **Class-imbalance handling** at the head (loss reweighting, calibrated thresholds, resampling) for the shock step, where only ~24 illicit nodes exist.
- 2. **Depth-robustness check**: Broadcast Bias is ruled out only at $K=1$; if deep propagation is used, re-test separability of the deeper embeddings before assuming they survive.
- 3. **Recovery adaptation**: For $\tau\ge44$, prefer mechanisms that down-weight stale topology (exponential decay of multiscale features, walk-forward retraining) over static deep aggregation.
