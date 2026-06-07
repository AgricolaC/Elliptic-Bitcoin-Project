# Course Methods Comparison & Defense Narrative

This document provides a structured theoretical defense of the Elliptic Bitcoin anomaly detection pipeline, contrasting our approach with the canonical methodologies covered in the course. This narrative serves as the foundation for the "Critical Discussion" and "Baselines" sections of the final report and oral exam.

## 1. Connection to Module 2: Geometric Learning & MPNNs
Our Simplified Graph Convolution (SGC) pipeline connects directly to the Message Passing Neural Network (MPNN) framework from Module 2. However, rather than learning adaptive message passing weights via GCNConv or GAT, we decouple the spatial aggregation from the feature transformation (following the SIGN/SGC paradigm). 
- **Theoretical Justification**: In homophilous anomaly detection tasks like Elliptic, pre-computing the Laplacian propagation $D^{-1/2} A D^{-1/2} X$ over multiple scales ($k=1,2,3$) acts as a low-pass filter over the graph manifold. This captures the local geometric structure without suffering from the over-smoothing and optimization instability that plagues end-to-end deep GCNs.
- **Intrinsic Dimensionality**: The raw 166 features have an intrinsic geometric structure (as discussed in Module 1 & `nb3b`). Our addition of handcrafted topology features (Sweep 4a: PageRank, Clustering Coefficients) explicitly maps structural roles that the spectral smoothing alone misses, providing the MLP head with orthogonal dimensions of variation.
- **Gauge-Theoretic Interpretation (`nb4 gauge theories exec.py`)**: As demonstrated in the course's gauge theory notebook, naive aggregation on a manifold violates gauge equivariance because aggregated features leave the local tangent plane. While standard GNNs on Euclidean spaces bypass this issue, the Elliptic DAG's locally-pooled neighborhood features implicitly encode a discrete gauge choice. 
- **Symmetric Normalization as a Connection**: Our use of the symmetrically normalized adjacency matrix $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ in SGC acts as the discrete analogue to choosing the Levi-Civita connection (the unique torsion-free connection compatible with the graph metric). This symmetric normalization is equivalent to assigning edge weights that locally flatten the discrete curvature, ensuring that the propagation phase natively preserves the structural gauge without requiring complex learnable per-layer transformations.

## 2. Comparison to DOMINANT (Module 4)
*Course Reference: `lecture3 dominant.py`, `nb5_gans_graphs.ipynb`*

Vaccarino's implementation of DOMINANT provides the canonical unsupervised approach to structural anomalies. It relies on a 2-layer GCN encoder and dual decoders (attribute and structure), optimizing a reconstruction objective where the anomaly score is $s(v) = \alpha ||x - \hat{x}||^2 + (1-\alpha) ||A - \hat{A}||^2$.

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
- **Our Baselines**: We explicitly benchmark against XGBoost and RandomForest over the raw attributes ($X \in \mathbb{R}^{166}$). This proves that any gains achieved by our SGC pipeline are strictly attributable to the geometric structure (message passing and topology features) rather than just the non-linear capacity of the classifier head.

## Conclusion: The Core Bottleneck
In alignment with the student elaborato template (Section 8), the central challenge of the Elliptic dataset is not just class imbalance, but **temporal generalization**. Our extensive walk-forward ablation matrix proves that while topological features improve static detection, the fundamental bottleneck remains the non-stationary "concept drift" over the 49 timesteps. Our decoupled SGC design offers the most robust response to this drift by allowing the decision boundary to adapt dynamically across time steps without requiring expensive GCN re-training.
