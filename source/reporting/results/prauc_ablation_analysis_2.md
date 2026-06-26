# Static Grid Analysis: The Marginal Effects on PR-AUC (Isolated by Variation)

Based on your strategic decision to center the project's reporting around PR-AUC, we have isolated the marginal effects of every architectural decision across the entire SGC Grid Sweep. 

Crucially, **PCA Compression** and the **Base (Uncompressed)** architectures interact with the graph topology in fundamentally different ways. By splitting our analysis between these two variations, the exact mechanisms of concept drift survival become clear.

> [!NOTE]
> **Metric Guide:**
> - **Validation:** The perfectly stable, pre-shock period. High scores here indicate strong learning capacity but don't guarantee resilience.
> - **Static OOT:** The volatile holdout set containing the massive $\tau=43$ shock. High scores here indicate robustness against adversarial concept drift.

---

## 1. The Impact of Propagation Depth (K)

| Variation | K | Val Pooled PR-AUC | Val Macro PR-AUC | OOT Pooled PR-AUC | OOT Macro PR-AUC |
|:---:|:-:|:---:|:---:|:---:|:---:|
| **Base** | 1 | 0.8915 | 0.8416 | 0.3004 | 0.2414 |
| **Base** | 2 | **0.9019** | **0.8477** | **0.3684** | **0.2761** |
| **Base** | 3 | 0.8956 | 0.8377 | 0.3671 | 0.2713 |
| --- | --- | --- | --- | --- | --- |
| **PCA** | 1 | 0.9012 | 0.8352 | 0.2766 | 0.2312 |
| **PCA** | 2 | 0.9146 | 0.8359 | 0.3391 | 0.2704 |
| **PCA** | 3 | **0.9092** | **0.8391** | **0.3909** | **0.3015** |

**Takeaway: Compression requires depth; Raw features do not.**
For the uncompressed **Base** features, expanding the neighborhood beyond 2 hops actually causes slight oversmoothing and performance degradation in OOT testing. 
However, if the node features are compressed into rigid principal components (**PCA**), the model relies heavily on the structural geometry of the graph to survive the $\tau=43$ shock. PCA models see massive gains scaling from $K=2$ to $K=3$ (OOT Pooled +0.051).

---

## 2. The Impact of Edge Directionality (Dir)

| Variation | Direction | Val Pooled PR-AUC | Val Macro PR-AUC | OOT Pooled PR-AUC | OOT Macro PR-AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base** | **F (Undirected)** | 0.8824 | **0.8432** | 0.3307 | **0.2638** |
| **Base** | **T (Directed)** | **0.9102** | 0.8414 | **0.3599** | 0.2620 |
| --- | --- | --- | --- | --- | --- |
| **PCA** | **F (Undirected)** | 0.8977 | 0.8305 | 0.3172 | 0.2684 |
| **PCA** | **T (Directed)** | **0.9189** | **0.8429** | **0.3538** | 0.2670 |

**Takeaway: Directionality universally improves Pooled OOT generalization.**
Regardless of whether features are compressed or raw, restricting the graph to true downstream financial flow (`Dir=T`) provides a ~0.03 to ~0.04 boost to the Pooled OOT PR-AUC. Undirected graphs (`Dir=F`) slightly preserve Macro stability for Base models, but sacrifice too much overall anomaly detection power.

---

## 3. The Impact of Topology Injection

| Variation | Injection | Val Pooled PR-AUC | Val Macro PR-AUC | OOT Pooled PR-AUC | OOT Macro PR-AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base** | **None** | 0.8864 | 0.8337 | 0.3233 | 0.2506 |
| **Base** | **Early** | 0.8976 | 0.8462 | **0.3601** | **0.2746** |
| **Base** | **Late** | **0.9049** | **0.8471** | 0.3526 | 0.2636 |
| --- | --- | --- | --- | --- | --- |
| **PCA** | **None** | 0.9065 | **0.8377** | 0.3351 | 0.2639 |
| **PCA** | **Early** | 0.9062 | 0.8356 | 0.3318 | 0.2684 |
| **PCA** | **Late** | **0.9123** | 0.8368 | **0.3397** | **0.2707** |

**Takeaway: PCA washes out topological injection.**
When utilizing raw features (**Base**), injecting explicit structural knowledge like PageRank and Degree is critical. Injecting it `Early` (so the structural metrics themselves are passed through the graph convolution) leads to the highest OOT survival (0.3601 Pooled PR-AUC).
Conversely, if node features are highly compressed via **PCA**, early topology injection actually hurts the model, and even late injection provides negligible benefits. This implies PCA models rely almost entirely on the raw $K$-depth message passing to build their structural understanding.

---

### Final Blueprint 

By isolating the interactions, we now have two distinct theoretical optimums to pursue:

1. **The Raw Feature Heavyweight:** `Base, K=2, Dir=T, Topo=Early`
   - Relies on pure feature expression and explicit topological metrics. Avoids deep convolutions to prevent oversmoothing.
2. **The Structural Deep Net:** `PCA, K=3, Dir=T, Topo=Late`
   - Relies heavily on deep 3-hop topological message passing to compensate for linearly compressed, rigid feature vectors.
