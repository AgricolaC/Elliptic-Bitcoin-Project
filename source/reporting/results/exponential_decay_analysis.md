## Exponential Decay Temporal Analysis

> **Why we did this**: Standard Walk-Forward (WF) models treat all prior timesteps as equally relevant or rely solely on the model architecture to forget obsolete information. By injecting **Exponential Decay ($\lambda$)** into the multi-scale graph features, we explicitly down-weight older topological signatures, providing a mechanism to handle the extreme concept drift seen post-AlphaBay shutdown ($\tau \ge 43$).

We evaluated Exponential Decay across three decay rates:
- **$\lambda = 0.05$** (Slow decay: long-term memory)
- **$\lambda = 0.25$** (Medium decay)
- **$\lambda = 0.50$** (Fast decay: short-term memory)

### 1. The XGBoost Amplification

XGBoost is intrinsically a purely tabular, node-level classifier. However, by providing it with exponentially decayed multiscale graph features, we observe a significant amplification in its temporal robustness.

| Configuration | Baseline WF Macro F1 | Decay WF Macro F1 | Best $\lambda$ |
|---|---|---|---|
| **XGBoost** | $0.634$ | **$0.674$** | **$0.50$** |

*   **Insight**: XGBoost heavily benefits from *fast* decay ($\lambda=0.50$). Because tabular trees greedily split on the most informative features, providing rapidly decaying topological embeddings allows the model to leverage recent neighborhood structures without being anchored to the obsolete Pre-Shock ($\tau \le 42$) topologies. The fast decay acts as an explicit regularizer against the Topological Recovery Trap.

### 2. SGC: A New Graph Champion Emerges

Previously, our best graph model was the highly rigid `SGC + MLP + MP (K=2, Dir=F, Topo=early)`, peaking at $0.489$ Macro F1. Exponential decay destabilizes this rigidity and crowns a new optimal configuration.

| Configuration | Baseline WF Macro F1 | Decay WF Macro F1 | Best $\lambda$ |
|---|---|---|---|
| **SGC (K=2, Dir=F, Topo=early)** | $0.489$ | $0.465$ | $0.50$ |
| **SGC (K=2, Dir=T, Topo=late)** | $0.437$ | **$0.527$** | **$0.25$** |
| **SGC (K=2, Dir=T, Topo=None)** | $0.442$ | **$0.501$** | **$0.25$** |
| **SGC (K=3, Dir=F, Topo=late)** | $0.472$ | $0.490$ | $0.50$ |
| **SGC (K=3, Dir=T, Topo=early)** | $0.446$ | $0.479$ | $0.50$ |
| **SGC (K=3, Dir=T, Topo=None)** | $0.441$ | $0.477$ | $0.25$ |

*   **Insight**: The `Dir=T` (Directed) and `Topo=late` configuration typically suffered from severe overfitting in standard WF due to the massive dimensionality and complexity of tracking exact directed paths post-message passing. 
*   **The Decay Synergy**: When Exponential Decay ($\lambda=0.25$) is applied, it "softens" these complex directed embeddings. The medium decay rate provides balance—it remembers enough of the darknet market structures to detect stable fraud rings, but forgets fast enough to avoid catastrophic failure when AlphaBay shuts down. This synergy boosts its performance by nearly $+0.09$ Macro F1, pushing SGC over the $0.50$ barrier for the first time. 

### 3. Timestep-by-Timestep Post-Shock Analysis

To understand how the models survive and recover from the AlphaBay shock, the table below tracks the Walk-Forward Macro F1 score timestep by timestep starting at the shock ($\tau=43$):

| Timestep ($\tau$) | Era / Regime | WF Champion SGC | Decay Champion SGC | XGBoost WF | XGBoost + decay |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$\tau=43$** | **Shock** | $0.0000$ | $0.1176$ | $0.0000$ | **$0.1538$** |
| **$\tau=44$** | **Recovery** | $0.0421$ | $0.0377$ | $0.3030$ | **$0.4000$** |
| **$\tau=45$** | **Recovery** | $0.0000$ | $0.0000$ | $0.1111$ | **$0.1600$** |
| **$\tau=46$** | **Recovery** | $0.0000$ | $0.4000$ | $0.4000$ | **$0.5000$** |
| **$\tau=47$** | **Recovery** | $0.0000$ | $0.0714$ | $0.3529$ | **$0.4390$** |
| **$\tau=48$** | **Recovery** | $0.5043$ | $0.5203$ | $0.4643$ | **$0.7397$** |
| **$\tau=49$** | **Recovery** | $0.5063$ | $0.7255$ | $0.7273$ | **$0.7826$** |
| **Mean** | **Post-Shock ($\tau \ge 44$)** | $0.1754$ | $0.2925$ | $0.3931$ | **$0.5036$** |

* **Shock Survival ($\tau=43$)**: Standard models (WF Champion SGC and XGBoost WF) collapse entirely to $0.0000$ Macro F1 immediately after the shock. Augmenting the feature representations with exponential decay keeps both models active, with XGBoost + decay leading at $0.1538$ Macro F1 and Decay Champion SGC achieving $0.1176$ Macro F1.
* **Rapid Recovery**: Decaying obsolete topologies allows the graph configurations to unlearn stale patterns. Decay Champion SGC rebounds to $0.4000$ Macro F1 at $\tau=46$ (matching the base XGBoost), whereas the non-decay WF Champion SGC remains frozen at $0.0000$ F1 until $\tau=48$.

### Conclusion

Exponential Decay successfully reduces the primary vulnerability of Graph Neural Networks in highly dynamic adversarial environments: **Structural Overfitting**. 
By explicitly forcing the topological features to decay, we unlocked a new graph-based champion (`K=2, Dir=T, Topo=late, λ=0.25` at $0.527$ Macro F1) and pushed the overall benchmark ceiling even higher with XGBoost (`λ=0.50` at $0.674$ Macro F1).
