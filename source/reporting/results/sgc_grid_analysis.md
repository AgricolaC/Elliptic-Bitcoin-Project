# SGC + MLP Grid Search Analysis

This report analyzes the hyperparameter grid search for the Simplified Graph Convolution (SGC) with an MLP head. We evaluate the effects of **Neighborhood Depth ($K$)**, **Graph Directionality ($Dir$)**, **Topological Features ($Topo$)**, and **Dimensionality Reduction ($PCA$)** on Out-of-Time (OOT) generalization.

## 1. Which combinations increase the scores?

The baseline undirected SGC+MLP (`K=1, Dir=F, Topo=None`) achieves an OOT Pooled F1 of **$0.321$**. 

The most successful configurations follow two distinct paths depending on the feature set:
* **Best Configuration (Base Features)**: `K=2, Dir=F, Topo=early` $\rightarrow$ **$0.455$** Pooled F1.
* **Best Configuration (PCA Features)**: `K=3, Dir=T, Topo=None` $\rightarrow$ **$0.477$** Pooled F1.

## 2. Does Directionality ($Dir=T$) Help?

**Yes, but it acts as a substitute for explicit topology.**
Setting $Dir=T$ means the graph convolution respects the directed nature of the Bitcoin transaction network (separating upstream vs. downstream money flow).
* **When $Topo=None$**: Directionality is highly beneficial. For example, at $K=3$, treating the graph as directed ($Dir=T$) rather than undirected ($Dir=F$) boosts Pooled F1 from $0.268$ to $0.401$. The separated flow of money acts as an implicit structural/topological signal!
* **When $Topo$ is present**: The benefit vanishes or becomes negative. For instance, at $K=2, Topo=early$, changing from undirected to directed drops performance from $0.455$ to $0.422$. This implies that explicit topological features (`early/late`) already capture the structural motifs. Forcing the GNN to also separate in/out edges causes redundancy and likely overfits to the pre-$\tau=43$ network structure.

## 3. What are the effects of higher $K$?

* **$K=1$ is too shallow**: It consistently underperforms across all configurations. The 1-hop neighborhood is insufficient to capture the broader illicit motifs.
* **$K=2$ is the sweet spot for Base features**: It provides the highest peaks without overfitting (e.g., $0.455$ with $Topo=early$ and undirected edges).
* **$K=3$ suffers from Oversmoothing (in Base)**: When pushing to 3 hops with the raw Base features, performance often collapses. For example, `K=3, Dir=F, Topo=None` plummets to $0.268$. The nodes' features become mathematically indistinguishable from their neighbors (the classic GNN oversmoothing problem).

## 4. When is PCA useful?

PCA provides a fascinating interaction with $K$.
* **At $K=1$ and $K=2$**: PCA almost universally **hurts** performance. The model needs the full expressiveness of the raw Base features, and PCA discards too much critical discriminative variance. For instance, `K=2, Dir=F, Topo=early` drops from $0.455$ (Base) to $0.357$ (PCA).
* **At $K=3$**: PCA becomes the **savior**. Because 3-hop aggregation causes severe feature noise and oversmoothing, PCA acts as a powerful regularizer. 
  * `K=3, Dir=F, Topo=None`: PCA boosts performance from $0.268$ to $0.437$.
  * `K=3, Dir=T, Topo=None`: PCA hits the **absolute maximum of the entire grid** at **$0.477$** Pooled F1 and **$0.270$** Macro F1.

## 5. Mathematical Proof of Oversmoothing (Intrinsic Dimensionality)

To mathematically verify *why* $K=3$ fails with raw features but succeeds with PCA, we analyzed the Intrinsic Dimensionality (ID) of the generated embeddings.

* **The Expansion Phase ($K=1 \rightarrow K=2$)**:
  * As message passing deepens from 1-hop to 2-hops, the ID expands (e.g., $7.52 \rightarrow 8.07$ for `Dir=F, Topo=None`). The model successfully gathers new, discriminative variance from the neighborhood.
* **The Oversmoothing Collapse ($K=3$)**:
  * At 3-hops, the ID abruptly collapses. For instance, `K=3, Dir=T, Topo=None` plummets to an ID of **$7.32$**. The features become indistinguishable (oversmoothed) and the F1 score drops.
* **The PCA Rescue**:
  * Applying PCA to that exact $K=3$ model compresses the ID down to **$6.59$** (the lowest ID of the entire grid). By mathematically forcing the network to discard the low-variance oversmoothed noise and retain only the principal components, PCA acts as the regularizer, boosting the F1 score of most configurations.

## Conclusion & Best Practices

1. **If using Raw Features (Base)**: Stick to $K=2$. Use explicit topological features ($Topo=early$) on an **undirected** graph ($Dir=F$).
2. **If forced to use Deep Neighborhoods ($K=3$)**: You **must** apply PCA to prevent oversmoothing. Combined with **Directed** edge propagation ($Dir=T$), this PCA-regularized deep neighborhood yields the highest Out-of-Time performance of any graph configuration tested.
