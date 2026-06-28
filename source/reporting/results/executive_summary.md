## Executive Summary

The Elliptic Bitcoin dataset presents a challenging anomaly detection task under temporal non-stationarity. Our analysis of the transaction graph reveals five core findings regarding model performance and the nature of the network shock:

1. **The Graph Structure:** The Elliptic graph is a sequence of directed transaction snapshots $G_\tau=(V_\tau,E_\tau)$. The macroscopic structure of the network (volume, density, mean degree) remains stable throughout the entire timeline.
2. **The Nature of the Shock ($\tau=43$):** The catastrophic failure of machine learning models post-shock is not caused by a sudden geometric collapse of the feature space. Instead, it is driven by a massive **class-prior collapse**.
3. **Graph Feature Learning:** Techniques like Simplicial Graph Convolution (SGC) propagation and PCA uncover useful graph structures and homophily before the shock. However, these graph features are highly sensitive to specific micro-motifs.
4. **Topological Overfitting:** Deep graph models overfit to the local transaction motifs of the pre-shock illicit economy. When illicit actors return in the recovery phase ($\tau \ge 44$) using different transaction patterns, the graph models fail to recognize them.
5. **The Tabular Advantage:** Ultimately, tabular models (such as XGBoost) remain the strongest overall benchmark we tested. Node-level tabular features (like raw transaction metadata) survive the regime change much better than aggregated neighborhood graph motifs. While the final MLP-head experiment (LayerNorm + SiLU) improved graph-specific performance, it did not surpass the best tabular baseline.

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

The supervised learning task is highly imbalanced: the positive (illicit) class represents a very small minority of the global distribution (typically $<10\%$ pre-shock, and $<1\%$ post-shock). Because the negative class vastly outnumbers the positive class, **OOT Macro PR-AUC** (for static comparisons) and **Walk-Forward Macro Illicit-F1** (for regime tracking) are the primary metrics for evaluating model performance.
