# Elliptic Bitcoin Dataset: Structural Schema & Tensor Alignment

All structural transformations, batching routines, and data pipelines in this project must strictly conform to the topological and memory constraints outlined below. Failure to adhere to this schema will result in temporal data leakage or Out-Of-Memory (OOM) hardware failures.

## 1. Graph Specification & Topology
* **Graph Type:** Directed Acyclic Graph (DAG) representing Bitcoin payment lineages and transactional flow.
* **Global Dimensions:** * Nodes ($|V|$): $203,769$ transactions.
  * Edges ($|E|$): $234,355$ directional payment flows.
* **Component Partitioning (Temporal Structure):** The global graph is divided into $49$ discrete, disconnected time-step graphs. 
  * **Crucial Constraint:** There are absolutely no edges that traverse temporal boundaries (i.e., a transaction at $t=3$ cannot point to a transaction at $t=4$). 
  * Individual time steps contain bounded sub-graphs scaling between $1,000$ and $8,000$ nodes.

## 2. Sparse Memory Footprint (COO Format)
* **Dense Representation Hazard:** Constructing a dense global adjacency matrix $A \in \mathbb{R}^{|V| \times |V|}$ utilizing `float32` precision demands $\sim 154\text{ GB}$ of VRAM, which exceeds the limits of standard compute environments.
* **Sparse Tensor Enforcement:** Edge storage must be maintained exclusively using the Coordinate (COO) layout via a long integer index tensor of shape $[2, E]$. This compresses the structural memory footprint down to $\sim 2\text{ MB}$, enabling $O(E)$ gather-scatter operations in PyTorch Geometric.

## 3. Indexing & Mapping Contracts
* **Global-to-Local Vectorization:** Raw Bitcoin transaction identifiers (e.g., `23048593`) are non-contiguous integers. The preprocessing pipeline must map these global IDs into localized, contiguous index spaces for each temporal batch:
  $$\text{txId} \to \{0, 1, \dots, |V_t|-1\}$$
* **Edge Alignment:** The `source` and `target` rows of the edge index must be re-indexed relative to the localized node coordinates of their respective time step $t$. Failing to do this will cause index out-of-bounds exceptions during tensor multiplication.

## 4. Feature Matrix ($X$) Configuration
Each node is represented by a $166$-dimensional continuous feature vector.
* **Local Features (Indices 0-93):** Transaction-specific metrics including time step, number of inputs/outputs, transaction fee, and output volume.
* **Aggregated Features (Indices 94-165):** Non-local, one-hop structural information representing the maximum, minimum, standard deviation, and correlation coefficients of the neighbor transactions.
* *(Note: Explicit topological invariants calculated via NetworkX, such as PageRank, will be appended starting at index 166).*

## 5. Labels & Optimization Targets
This dataset poses a severe semi-supervised anomaly detection challenge characterized by extreme class imbalance.
* **Class Distribution:**
  * **Class 1 (Illicit Anomaly):** $4,545$ nodes ($\sim 2\%$).
  * **Class 2 (Licit Normal):** $42,019$ nodes ($\sim 21\%$).
  * **Unknown (Unlabeled):** $157,205$ nodes ($\sim 77\%$).
* **Masking Protocol:** "Unknown" nodes must be explicitly masked out or dropped during supervised loss calculation (e.g., Cross-Entropy Loss). However, their feature vectors and connections must remain in the adjacency tensor during the message-passing/SGC phase to preserve the structural geometry of the manifold.
* **Evaluation Metrics:** Due to the $2\%$ minority class, standard Accuracy is an invalid and misleading metric. Optimization engines and final evaluations must rely strictly on **Minority-Class F1-Score** and **Precision-Recall AUC**.