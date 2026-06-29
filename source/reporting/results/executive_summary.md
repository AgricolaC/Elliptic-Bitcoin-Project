## Executive Summary

The Elliptic Bitcoin dataset presents a challenging anomaly detection task under temporal non-stationarity. Our analysis of the transaction graph reveals five core findings regarding model performance and the nature of the network shock:

1. **The Graph Structure:** The Elliptic graph is a sequence of directed transaction snapshots $G_\tau=(V_\tau,E_\tau)$. The macroscopic structure of the network (volume, density, mean degree) remains stable throughout the entire timeline.
2. **The Nature of the Shock ($\tau=43$):** The catastrophic failure of machine learning models post-shock likely is not caused by a sudden geometric collapse of the feature space. Instead, it is driven by a massive **class-prior collapse**.
3. **Graph Feature Learning:** Techniques like Simplicial Graph Convolution (SGC) propagation and PCA uncover useful graph structures and homophily before the shock. However, these graph features are highly sensitive to specific micro-motifs.
4. **Topological Overfitting (Validated by $\tau=44$ MMD drift):** Deep graph models overfit to the local motifs of the pre-shock illicit economy. The compounding MMD jump at $\tau=44$ confirms that returning illicit actors exhibit fundamentally new transactional signatures; the graph models fail to generalize because they remain tethered to the now-obsolete local connectivity patterns of the pre-shock era.
5. **The Tabular Advantage Paradox (Feature Selection):** Ultimately, tabular models (such as XGBoost) remain the strongest benchmark. However, this is not because they ignore the graph (72 of the 166 dataset features are explicitly 1-hop and 2-hop graph aggregates). When Broadcast Bias and Subpopulation Shift smear the neighborhood distributions at $\tau=43$, these 72 aggregated features become corrupted. XGBoost survives because of **feature selection**: its decision trees can dynamically assign zero importance to the toxic graph features and rely entirely on the 94 purely local tabular features. A standard GCN, by architectural design, forces the blending of local and neighborhood features at every layer, fatally ingesting this topological noise. 

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
