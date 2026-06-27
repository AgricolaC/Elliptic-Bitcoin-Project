# Baseline Performance & Computational Efficiency

This report contrasts the tabular baselines (including IsolationForest, Logistic Regression, RandomForest, and XGBoost) against the graph-based baselines (GCN, SGC, and SGC + MLP Head) using the results from `sweep_results.csv` and `final_aggregated_results.csv`. The focus is on the Out-of-Time (OOT) generalization metrics (Pooled and Macro Illicit-F1) evaluated against computational training time.

## 1. Summary of Results

| Model | Training Time (s) | OOT Pooled F1 | OOT Macro F1 |
| --- | --- | --- | --- |
| **IsolationForest** | $0.003$ | N/A | N/A |
| **Logistic Regression** | $0.181$ | $0.228$ | $0.241$ |
| **SGC (Baseline)** | $1.456 \pm 0.33$ | $0.219 \pm 0.001$ | $0.212 \pm 0.001$ |
| **XGBoost** | $2.785$ | $0.765$ | $0.475$ |
| **RandomForest** | $6.764$ | **$0.777$** | **$0.479$** |
| **SGC + MLP Head** | $7.733 \pm 0.14$ | $0.455 \pm 0.042$ | $0.261 \pm 0.026$ |
| **PyG GCN (2-layer)** | $172.378$ | $0.480$ | $0.287$ |

*(Note: SGC and SGC+MLP metrics are averaged across seeds 42, 43, and 44 on the Base feature set).*

## 2. Analytical Insights

### The Computational Cost of Message Passing
The standard Graph Convolutional Network (GCN) is incredibly slow compared to all other methods. It takes **172.38 seconds** to train. In contrast, the tabular tree-based models (XGBoost and RandomForest) train in just **2.78s to 6.76s** (up to 60x faster). 

### The Power of Simplified Graph Convolutions (SGC)
SGC mathematically precomputes the neighborhood aggregation, removing the weight matrices between graph convolutional layers. This collapses the GNN into a simple feature preprocessing step followed by logistic regression.
* **Speed**: Training SGC takes only **~1.45 seconds**. This is over **100x faster** than the standard PyG GCN.
* **Performance Trap**: However, SGC *alone* effectively performs just like Logistic Regression. Its OOT Pooled F1 is $0.219$ (compared to LR's $0.228$). The linear classifier simply cannot capture the complex, non-linear illicit topologies.

### The Hybrid Compromise: SGC + MLP Head
By replacing the simple linear classifier in SGC with a Multi-Layer Perceptron (MLP), the model regains the ability to model non-linear boundaries on top of the aggregated graph features.
* **Speed**: Training takes **~7.73 seconds**. This adds a modest computational overhead over base SGC but is still **22x faster** than the full GCN.
* **Performance**: The OOT Pooled F1 doubles from $0.219$ to **$0.455$**, and it practically matches the full GCN's performance ($0.480$) at a fraction of the computational cost!

### The Dominance of Tree-Based Tabular Models
Despite the massive focus on graph neural networks for this dataset, the simple **RandomForest** and **XGBoost** models completely dominate the OOT metrics. 
* **Performance**: RandomForest achieves the peak Pooled F1 of **$0.777$**, with XGBoost close behind at **$0.765$**. Both vastly outperform the deep GCN ($0.480$).
* **Speed**: XGBoost is exceptionally fast (**$2.78s$**), making it over twice as fast as RandomForest ($6.76s$) and almost as fast as linear SGC, while delivering state-of-the-art results.

## 3. Conclusion
The results strongly suggest that the non-linear feature interactions (which Tree-Based models handle perfectly) are far more predictive for illicit transactions than deep, iterative message passing across the graph structure.

If graph features are strictly necessary, **SGC + MLP Head** provides 95% of the performance of a deep GCN at 4% of the computational cost. However, a well-tuned **XGBoost** or **RandomForest** tabular model currently remains the indisputable superior choice for speed, scalability, and out-of-time robustness in this network.
