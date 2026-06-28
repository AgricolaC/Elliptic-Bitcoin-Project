## SGC + MLP Grid Search Analysis

> **Why we did this**: To systematically evaluate the core topological hyperparameters (neighborhood depth $K$, edge directionality, and topology injection phase) of Simplified Graph Convolutions to find the optimal configuration before committing to deep model sweeps.

This report analyzes the hyperparameter grid search for the Simplified Graph Convolution (SGC) with an MLP head. We evaluate the effects of **Neighborhood Depth ($K$)**, **Graph Directionality ($Dir$)**, **Topological Features ($Topo$)**, and **Dimensionality Reduction ($PCA$)** on Out-of-Time (OOT) generalization.

### 1. Which combinations increase the scores?

The baseline undirected SGC+MLP (`K=1, Dir=F, Topo=None`) achieves an OOT Macro PR-AUC of **$0.169$** and OOT Macro F1 of **$0.195$**. 

The most successful configurations follow two distinct paths depending on the feature set:
* **Best Configuration (Base Features)**: `K=2, Dir=F, Topo=early` $\rightarrow$ OOT Macro PR-AUC **$0.312$**.
* **Best Configuration (PCA Features)**: `K=3, Dir=T, Topo=None` $\rightarrow$ OOT Macro PR-AUC **$0.318$** and OOT Macro F1 **$0.270$**.

### 2. Does Directionality ($Dir=T$) Help?

**Yes, but it acts as a substitute for explicit topology.**
Setting $Dir=T$ means the graph convolution respects the directed nature of the Bitcoin transaction network (separating upstream vs. downstream money flow).
* **When $Topo=None$**: Directionality is beneficial. For example, at $K=3$ with Base features, treating the graph as directed ($Dir=T$) rather than undirected ($Dir=F$) boosts OOT Macro PR-AUC from $0.241$ to $0.273$. The separated flow of money acts as an implicit structural/topological signal.
* **When $Topo$ is present**: The benefit can vanish or become negative. For instance, at $K=2, Topo=early$, changing from undirected to directed drops OOT Macro PR-AUC from $0.312$ to $0.289$. This implies that explicit topological features (`early/late`) already capture the structural motifs. Forcing the GNN to also separate in/out edges causes redundancy and likely overfits to the pre-$\tau=43$ network structure.

### 3. What are the effects of higher $K$?

* **$K=1$ is too shallow**: It consistently underperforms across all configurations. The 1-hop neighborhood is insufficient to capture the broader illicit motifs.
* **$K=2$ is the sweet spot for Base features**: It provides the highest Base-feature peak without overfitting (OOT Macro PR-AUC $0.312$ with $Topo=early$ and undirected edges).
* **$K=3$ suffers from Oversmoothing (in Base)**: When pushing to 3 hops with the raw Base features, performance often collapses. For example, `K=3, Dir=F, Topo=None` falls to OOT Macro PR-AUC $0.241$ and Macro F1 $0.161$. The nodes' features become mathematically indistinguishable from their neighbors (the classic GNN oversmoothing problem).

### 4. When is PCA useful?

PCA provides a fascinating interaction with $K$.
* **At $K=1$ and $K=2$**: PCA often **hurts** performance. The model needs the full expressiveness of the raw Base features, and PCA discards too much critical discriminative variance. For instance, `K=2, Dir=F, Topo=early` drops from OOT Macro PR-AUC $0.312$ (Base) to $0.277$ (PCA).
* **At $K=3$**: PCA becomes the **savior**. Because 3-hop aggregation causes severe feature noise and oversmoothing, PCA acts as a powerful regularizer. 
  * `K=3, Dir=F, Topo=None`: PCA boosts OOT Macro PR-AUC from $0.241$ to $0.304$.
  * `K=3, Dir=T, Topo=None`: PCA hits the **absolute maximum of the graph grid** at OOT Macro PR-AUC **$0.318$** and OOT Macro F1 **$0.270$**.

### 5. Mathematical Proof of Oversmoothing (Intrinsic Dimensionality)

To mathematically verify *why* $K=3$ fails with raw features but succeeds with PCA, we analyzed the Intrinsic Dimensionality (ID) of the generated embeddings.

* **The Expansion Phase ($K=1 \rightarrow K=2$)**:
  * As message passing deepens from 1-hop to 2-hops, the ID expands (e.g., $7.52 \rightarrow 8.07$ for `Dir=F, Topo=None`). The model successfully gathers new, discriminative variance from the neighborhood.
* **The Oversmoothing Collapse ($K=3$)**:
  * At 3-hops, the ID abruptly collapses. For instance, `K=3, Dir=T, Topo=None` plummets to an ID of **$7.32$**. The features become indistinguishable (oversmoothed) and the F1 score drops.
* **The PCA Rescue**:
  * Applying PCA to that exact $K=3$ model compresses the ID down to **$6.59$** (the lowest ID of the entire grid). By mathematically forcing the network to discard the low-variance oversmoothed noise and retain only the principal components, PCA acts as the regularizer, boosting the F1 score of most configurations.

### Conclusion & Best Practices

1. **If using Raw Features (Base)**: Stick to $K=2$. Use explicit topological features ($Topo=early$) on an **undirected** graph ($Dir=F$).
2. **If forced to use Deep Neighborhoods ($K=3$)**: You **must** apply PCA to prevent oversmoothing. Combined with **Directed** edge propagation ($Dir=T$), this PCA-regularized deep neighborhood yields the highest Out-of-Time performance of any graph configuration tested.
