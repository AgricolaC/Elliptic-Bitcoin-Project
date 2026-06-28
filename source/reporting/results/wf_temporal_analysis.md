## Walk-Forward (WF) Temporal Analysis

> **Why we did this**: Static OOT evaluation asks whether a model trained on early timesteps survives the future. Walk-forward evaluation asks a deployment-style question: if the model is repeatedly retrained with the newest available snapshot, can it adapt to the post-shutdown regime?

For the presentation, this section is standardized to **Macro** metrics:

* **WF Macro PR-AUC**: threshold-free ranking quality averaged across test timesteps; this is the primary plotted metric.
* **WF Macro F1**: fixed-threshold classification quality averaged across test timesteps; this is secondary table context.

We use Macro throughout this section so large snapshots cannot dominate small but important timesteps.

### 0. The transition from OOT to WF

When transitioning from Static OOT evaluation to continuous Walk-Forward training, models generally improve because the training window keeps moving forward. However, the ranking of graph configurations changes.

| Configuration | Static OOT Macro PR-AUC | WF Macro PR-AUC |
|---|---:|---:|
| **Grid: K=3, Dir=T, Topo=None (PCA)** | **$0.318$** | $0.452$ |
| **Grid: K=2, Dir=F, Topo=None (Base)** | $0.289$ | **$0.515$** |
| **Grid: K=3, Dir=F, Topo=early (Base)** | $0.287$ | $0.501$ |
| **Grid: K=3, Dir=F, Topo=late (PCA)** | $0.290$ | $0.508$ |
| **Grid: K=2, Dir=T, Topo=late (Base)** | $0.290$ | $0.495$ |

The static winner, `K=3, Dir=T, Topo=None` with PCA, is strong when it must extrapolate from a fixed early training period. In WF, though, the model is updated one snapshot at a time. The simpler `K=2, Dir=F, Topo=None` graph becomes more robust because it is less tied to exact directed micro-structure.

### 1. Macro-only WF summary

| Model | WF Macro F1 | WF Macro PR-AUC |
|---|---:|---:|
| **SGC baseline** | $0.309$ | $0.291$ |
| **SGC + MLP head** | $0.408$ | $0.447$ |
| **Grid: K=1, Dir=F, Topo=late** | $0.458$ | $0.499$ |
| **Grid: K=2, Dir=F, Topo=None** | $0.481$ | $0.515$ |
| **PCA: K=3, Dir=T, Topo=None** | $0.432$ | $0.452$ |
| **PCA: K=3, Dir=F, Topo=late** | $0.454$ | $0.508$ |
| **XGBoost WF** | $0.634$ | $0.713$ |

The best graph-only WF configuration is `K=2, Dir=F, Topo=None`, but the best overall WF model remains XGBoost by a large margin.

### 2. The shock is still visible

At $\tau=43$, every model is stressed by the sudden label/prior change. This supports the interpretation that the event is a **prior probability shift** rather than a simple embedding-collapse problem.

The important presentation point is not that graph models never recover; it is that their recovery depends heavily on avoiding brittle topological overfitting.

### 3. Temporal decay

The temporal decay ablation applies exponential downweighting to older samples during WF training. Under Macro metrics, decay is still useful:

| Model | No-decay WF Macro PR-AUC | Best-decay WF Macro PR-AUC | Best decay |
|---|---:|---:|---:|
| **Grid: K=2, Dir=T, Topo=early** | $0.452$ | **$0.586$** | $\lambda=0.25$ |
| **Grid: K=2, Dir=T, Topo=late** | $0.441$ | **$0.585$** | $\lambda=0.25$ |
| **XGBoost** | $0.713$ | **$0.720$** | $\lambda=0.25$ |

The graph gains are larger because decay directly attacks the graph model’s failure mode: memorizing stale pre-shutdown topology.

### 4. Conclusion

Walk-forward training improves graph models, but it does not overturn the main ranking. The most defensible final claim is:

> Temporal retraining and decay help SGC-style graph models adapt, but XGBoost remains the strongest overall model under WF Macro metrics.
