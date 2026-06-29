## Walk-Forward (WF) Temporal Analysis

> **Why we did this**: To understand how models respond to the temporal shock at $\tau=43$ when trained continuously. Static Out-of-Time evaluation is rigid; Walk-Forward training mimics real-world deployment, allowing us to see if models can 'recover' by learning the post-shock regime.

We analyze the Walk-Forward (WF) cross-validation results purely in terms of **Macro F1**. We segment the evaluation into three distinct temporal phases:
1. **Pre-Shock ($\tau \le 42$)**: The stable darknet market era.
2. **The Shock ($\tau = 43$)**: The AlphaBay shutdown event.
3. **Recovery ($\tau \ge 44$)**: The post-shutdown era where the graph topology drastically drifts.

### 1. The Transition from OOT to WF: The Graph Limit

When transitioning from Static Out-of-Time (OOT) evaluation to continuous Walk-Forward (WF) training, models generally see a noticeable performance increase. However, it is crucial to acknowledge that this boost is driven by two confounding factors:
1. **Recency (Adaptation):** WF allows the model to continuously train on the timesteps immediately preceding the test fold, adapting better to gradual temporal drift.
2. **Data Volume:** By the later folds (e.g., predicting $\tau=49$), WF trains on significantly more historical timesteps ($1 \le \tau \le 48$) compared to the Static OOT baseline (which strictly freezes the training set at $\tau \le 26$). 

Despite this dual advantage in data availability, we still observe a hard ceiling for Graph models.

| Configuration | Static OOT Macro F1 | Walk-Forward Macro F1 |
|---|---|---|
| **XGBoost (Tabular Baseline)** | **$0.475$** | **$0.634$** |
| **Grid: K=2, Dir=F, Topo=early (Base)** | $0.322$ (SGC Optimum) | $0.489$ (SGC Optimum) |
| **Grid: K=3, Dir=F, Topo=late (Base)** | $0.260$ | $0.472$ |
| **Grid: K=2, Dir=F, Topo=None (Base)** | $0.240$ | $0.466$ |
| **IPCA: K=3, Dir=T, Topo=None** | $0.282$ | $0.441$ |
| **Grid: K=3, Dir=T, Topo=early (Base)** | $0.260$ | $0.446$ |

#### The Tabular Ceiling is Still Unbreakable
While the optimal SGC graph configuration (`K=2, Dir=F, Topo=early`) successfully maintains its rank as the best Graph model across both static and continuous settings ($0.322 \rightarrow 0.489$), it still fundamentally fails to compete with tabular tree models. 

### 2. Baseline Temporal Performance Summary

By breaking down the Walk-Forward performance into discrete temporal phases, we can identify exactly *where* and *why* structural graph models fail compared to tabular baselines.

| Model | WF Macro F1 | Pre-Shock (1-42) Macro F1 | Shock (43) Macro F1 | Recovery (44-49) Macro F1 |
|---|---|---|---|---|
| **XGBoost WF** | **$0.634$** | **$0.895$** | $0.000$ | **$0.393$** |
| **SGC + MLP + MP (K=3, Dir=T, Topo=late)** | $0.442$ | $0.687$ | $0.000$ | $0.189$ |
| **SGC + MLP + MP (K=2, Dir=F, Topo=early)** | $0.489$ | $0.786$ | $0.000$ | $0.175$ |
| **SGC + MLP + MP (K=3, Dir=T, Topo=early)** | $0.446$ | $0.706$ | $0.000$ | $0.174$ |
| **SGC + MLP + MP (K=3, Dir=F, Topo=late)** | $0.472$ | $0.756$ | $0.000$ | $0.173$ |
| **SGC + MLP + MP (K=2, Dir=F, Topo=None)** | $0.466$ | $0.745$ | $0.000$ | $0.171$ |
| **SGC + MLP (K=3, Dir=F, Topo=None)** | $0.387$ | $0.549$ | $0.000$ | $0.235$ |
| **SGC + MLP (K=2, Dir=F, Topo=early)** | $0.413$ | $0.619$ | $0.000$ | $0.207$ |
| **SGC (K=2, Dir=F, Topo=None)** | $0.309$ | $0.480$ | $0.016$ | $0.130$ |

### 3. Analytical Insights on Baselines

#### The $\tau=43$ Collapse is Universal
Every baseline model—whether graph-based or tabular—experiences a catastrophic failure at $\tau=43$. XGBoost and the vast majority of SGC configurations completely fail to identify any illicit transactions ($0.000$ Macro F1). This corroborates that the AlphaBay shutdown might be a **Prior Probability Shift** (extreme label deprivation) rather than purely a geometric or feature failure. The models simply do not have the statistical confidence to predict the minority class when its base rate collapses so abruptly. Only IPCA (`K=3, Dir=T, Topo=None`) maintains a negligible pulse ($0.075$).

#### The Graph Recovery Trap
In the **Recovery** phase ($\tau \ge 44$), the discrepancy between Graph Models and Tabular Models explodes.
* **XGBoost** rebounds, although mildly, establishing a $0.393$ Macro F1 in the post-shock regime. By relying on purely local node-level features, it easily adapts to the new market dynamics.
* **SGC Models** plunge into the *Topological Recovery Trap*. The best graph architectures plummet from ~$0.78$ pre-shock down to $0.17$ during recovery. By baking the neighborhood topology deeply into the node features via message passing, Graph Neural Networks overfit to the pre-shutdown network structure. When that massive structural drift occurs post-$\tau=43$, they are rendered practically useless because the multi-hop geometries they memorized no longer exist. 

#### Regularization via Scale-Collapse (NoMP)
Interestingly, the models that recover the best among the graph architectures are those that omit Multiscale Propagation entirely. `SGC + MLP (NoMP, K=3, Dir=F, Topo=None)` achieves a $0.235$ Recovery Macro F1, noticeably higher than the heavily aggregated Multiscale optima ($\sim0.175$). 
When Multiscale Propagation (MP) is used, the feature tensor concatenates all neighborhood scales ($X \parallel AX \parallel \ldots$), allowing the MLP to learn highly complex interactions between different hop levels. This causes the model to overfit to the exact multi-scale structural signatures of the pre-shock graph. By omitting MP, the model is forced to use *only* the single, final smoothed representation ($A^K X$). Collapsing the topology into a single scale acts as a massive regularizer—the MLP has fewer parameters and cannot learn complex inter-scale interactions, allowing it to adapt more cleanly to the post-shock reality.

### Conclusion
Walk-Forward continuous training reveals the fatal flaw of deep topological aggregation in highly dynamic systems. While Graph models (like SGC+MLP+MP) perform adequately in the stable Pre-Shock regime ($~0.78$ Macro F1), their structural rigidity causes them to overfit to outdated topologies, preventing recovery after major systemic shocks ($~0.17$ Macro F1). XGBoost escapes this trap relatively better, establishing itself as the champion of temporal robustness in the walk-forward tests ($0.393$ Recovery Macro F1).
