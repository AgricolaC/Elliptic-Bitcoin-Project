# Course Methods Comparison & Defense Narrative

This document provides a structured theoretical defense of the Elliptic Bitcoin anomaly detection pipeline, contrasting our approach with the canonical methodologies covered in the course. This narrative serves as the foundation for the "Critical Discussion" and "Baselines" sections of the final report and oral exam.

## 1. Connection to Module 2: Geometric Learning & SIGN-lite Architectures
Our architecture is essentially a specialized implementation of **SIGN** (Scalable Inception Graph Neural Networks). By decoupling the spatial aggregation from the feature transformation, we pre-compute the heavy geometric lifting ($S^k X$).
- **The Feature Injection Strategy (The Sweet Spot)**: As established in high-tier benchmarking literature, the true power of decoupled models like SIGN lies in feature injection. By pre-computing the spectral smoothing, we create the perfect architectural blueprint to append complex geometric priors without massive memory overhead. Our **Sweep 4** perfectly embodies this strategy: we extract handcrafted topological features (PageRank, Clustering Coefficients) and concatenate them directly into the pre-computed multiscale representations. This hybrid approach bridges the gap—granting the MLP the expressive power of deep structural analysis while maintaining the execution speed of a lightweight baseline.
- **Intrinsic Dimensionality**: The raw 166 features have an intrinsic geometric structure. Our addition of graph topology features explicitly maps structural roles that the spectral smoothing alone misses, providing the MLP head with orthogonal dimensions of variation.
- **Gauge-Theoretic Interpretation (`nb4 gauge theories exec.py`)**: As demonstrated in the course's gauge theory notebook, naive aggregation on a manifold violates gauge equivariance because aggregated features leave the local tangent plane. While standard GNNs on Euclidean spaces bypass this issue, the Elliptic DAG's locally-pooled neighborhood features implicitly encode a discrete gauge choice. 
- **Symmetric Normalization as a Connection**: Our use of the symmetrically normalized adjacency matrix $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ in SGC can be interpreted as analogous to choosing the Levi-Civita connection (a torsion-free connection compatible with a metric). This symmetric normalization acts like assigning edge weights that locally flatten the discrete curvature, providing a canonical, basis-independent choice of propagation operator that reliably preserves structural gauge without requiring complex learnable per-layer transformations.


## 2. Comparison to DOMINANT (Module 4)
*Course Reference: `lecture3 dominant.py`, `nb5_gans_graphs.ipynb`*

The implementation of DOMINANT provides the canonical unsupervised approach to structural anomalies. It relies on a 2-layer GCN encoder and dual decoders (attribute and structure), optimizing a reconstruction objective where the anomaly score is $s(v) = \alpha ||x - \hat{x}||^2 + (1-\alpha) ||A - \hat{A}||^2$.

- **Contrast**: Our pipeline is fundamentally **supervised and discriminative**, optimized for F1-score via a logistic bottleneck, whereas DOMINANT is **unsupervised and generative** (reconstruction-based).
- **Trade-offs**: DOMINANT's reconstruction error is highly sensitive to the capacity of the bottleneck (as explored in `nb3_autoencoders`). If the intrinsic dimensionality of illicit nodes overlaps with normal nodes, reconstruction error fails to separate them. Because the Elliptic dataset provides ground-truth labels per timestep, a discriminative approach (like our MLP head) is empirically stronger for defining the precise decision boundary. Our architecture deliberately trades the label-free flexibility of DOMINANT for the precision of supervised F1 optimization.

## 3. Comparison to LSTM-AE (Module 3)
*Course Reference: `lstm_ae_notebook.ipynb`, `nb4_lstm_ae.ipynb`*

The LSTM-AE demonstrates temporal anomaly detection via sequence-to-sequence reconstruction, measuring the train/test reconstruction gap caused by concept drift.

- **Contrast**: The LSTM-AE assumes a continuous, uniformly sampled multivariate time series where a sliding window can capture temporal causality. The Elliptic dataset consists of 49 discrete, temporally disjoint subgraphs.
- **Trade-offs**: Applying a sequence model like LSTM over the disjoint subgraphs introduces false temporal continuity assumptions. Instead, our **Walk-Forward Validation** explicitly models the temporal dimension as "concept drift." By evaluating an expanding/sliding training window, we respect the strict causal boundary ($max(train) < \tau$) without forcing a recurrent architecture onto disjoint event graphs. The performance gap between our Static OOT and Walk-Forward Mean metrics directly quantifies the "evolving normality" and structural drift observed in the dataset.

## 4. Addressing Classical Baselines
*Course Reference: `nb1_foundations.ipynb`*

The taxonomy of classical methods (LOF, Isolation Forest, One-Class SVM) provides the baseline for anomaly detection. 
- **Scalability**: Algorithms like LOF are $O(n^2)$ distance-based, rendering them computationally prohibitive over the 203K nodes of the Elliptic dataset.
- **Our Baselines**: We explicitly benchmark against XGBoost and RandomForest over the raw attributes ($X \in \mathbb{R}^{166}$). 

## 5. The "Catch-22" of Multicollinearity and the Failure of PCA
When extending the architecture to include Directional Channels (In/Out/Undirected), the feature dimension explodes to 1176. This creates a severe case of multicollinearity. We evaluated two methods to combat this dimensional bloat:

1. **Heavy L2 Regularization (`5e-3`)**: While a massive L2 penalty prevents the MLP from memorizing noise, it simultaneously starves the model's capacity over 200 epochs, capping the directional model's performance at F1 `0.621` (Sweep 5).
2. **PCA Dimensionality Reduction**: We applied PCA (retaining 99% variance) to compress the propagated features. 
   - **Undirected Model (Sweep 4) (PCA 99%)**: F1 plummeted from `0.721` to `0.650`.
   - **Directional Model (Sweep 5) (PCA 99%)**: Compressed from 1176 to 339 dims, but F1 fell to `0.558`.

3. **Supervised Random Forest Feature Selection**: Unlike PCA, Random Forest Gini importance explicitly measures how well a feature separates illicit nodes from licit ones. We calculated the Gini importances of the directional model, sorted them, and kept only the subset of features needed to reach 99% cumulative predictive importance.
   - **Directional Model (RF 99%)**: Selected 886 out of 1176 features. F1 surged from `0.621` up to **`0.711`** in Sweep 5!

**Theoretical Conclusion**: Unsupervised PCA fails because the fraud signal is mathematically quiet. *Supervised Feature Selection* (RF) successfully identifies and drops the multicollinear noise, rescuing the Directional Model (Sweep 5) from `0.621` up to `0.711`. 

**However, the ultimate takeaway is that Sweep 4 remains supreme.** Even with advanced supervised feature selection, Sweep 5 (`0.711`) still falls short of the natural, unpruned Undirected Model (Sweep 4 at `0.721`). The directional channels introduce massive noise without providing any orthogonal predictive power that Sweep 4 hasn't already captured cleanly. 

## Conclusion: Tabular vs Topological Concept Drift
In alignment with the student elaborato template (Section 8), the central challenge of the Elliptic dataset is not just class imbalance, but **temporal generalization**. 

Our Walk-Forward evaluations revealed a critical distinction:
1. **Raw XGBoost** achieves a highly robust `0.871` Walk-Forward F1. It mostly survives the Step 43 dark market shutdown because it relies purely on tabular node attributes (which experienced only moderate drift).
2. **Sweep 4 (SGC + MLP)** achieves `0.721` on static data, but drops to `0.625` in Walk-Forward. 
3. **Hybrid Sweep 6 (SGC + XGBoost)**: To isolate whether the MLP or the Graph Structure was to blame, we fed the rich topological SGC features into an XGBoost head. The hybrid model achieved near-perfect fraud detection before the shutdown (e.g., F1 `0.970` at Step 35, `0.965` at Step 41). However, exactly at **Step 43**, the performance catastrophically collapsed to **`0.077`**.

**Final Thesis: The Structural Hysteresis Paradox** 
The Elliptic dataset's dark market shutdown wasn't just a tabular shift; it was a fundamental **Topological Concept Drift** (a phase transition of the underlying graph manifold). 
* **Tabular Elasticity:** Baseline XGBoost survived the recovery phase because its local features are structurally elastic. When tabular behavior changes, the decision trees quickly establish new boundaries and recover.
* **Topological Rigidity & Hysteresis:** Because our decoupled SIGN model mathematically captures the pre-shutdown global topology perfectly, it becomes fatally over-indexed on an obsolete graph structure when the network re-wires into sparse peel-chains. The model suffers from **Structural Hysteresis**—it is permanently biased by the "ghost" of the old network structure, rendering the geometric priors highly toxic post-shutdown.

Thus, while injecting structural priors (PageRank, Laplacian smoothing) yields phenomenal static results, it creates a brittle glass cannon under topological drift unless mitigated by strict memory bounding (sliding windows) or dynamic edge pruning.

---

Based on the empirical findings and theoretical limitations discovered during ablation, the defense should be structured around these three capstone insights:

### 1. The "Architectural Tension" Slide (DropEdge vs. SGC)
**The Premise:** To survive adversarial rewiring, a Graph Neural Network needs stochastic edges (DropEdge) to prevent overfitting to specific laundering routes. 
**The Tension:** Decoupled architectures like SGC and SIGN derive their speed by pre-computing $S^k X$ offline. If you introduce stochastic edges, the graph changes every epoch, destroying the pre-computation advantage and forcing the massive $O(L \cdot |E|)$ message-passing complexity back into the training loop. This fundamentally highlights the trade-off between architectural scalability and adversarial robustness.

### 2. The "Pragmatic Amputation" Slide (Sliding Window)
**The Premise:** How do we mitigate Structural Hysteresis without incurring the massive computational penalties of Reinforcement Learning or continuous model adaptation?
**The Solution:** The Sliding Window. We present the F1-trace showing the Hybrid model collapsing to $0.077$ and stagnating at $0.377$ under an expanding window. Then, we reveal the Sliding Window ($t-4$) ablation, showing the model aggressively amputating the toxic pre-shutdown geometry and violently snapping back to $0.893$ F1. 

### 3. The "Future Work" Slide (TDA & Betti-1 Cycles)
**The Premise:** Global topological invariants (like PageRank) are inherently brittle because they rely on macro-level stability. 
**The Frontier:** The mathematical future of adversarial graph learning is localized **Topological Data Analysis (TDA)**. By extracting persistent homology (Betti-1 cycles) from local $k$-hop ego-networks, we can identify laundering "peel-chains" as invariant topological holes. A cycle is a cycle, whether it occurs in a massive centralized dark market or a highly fractured decentralized tumbler. This localized geometric anchor survives macro-level concept drift.
