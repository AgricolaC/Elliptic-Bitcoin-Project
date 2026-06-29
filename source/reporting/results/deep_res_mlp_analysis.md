## Deep MLP Head Ablation: Residuals vs LayerNorm-SiLU

> **Why we did this**: We wanted to test whether a stronger classifier head on top of multiscale SGC propagation could improve graph-model performance. The original hypothesis emphasized a deep residual MLP, but the experiments show a more nuanced result: **residual connections did not help**. The useful head was small, normalized, and non-residual.

This report summarizes the individual head trials and the four sweep phases in `results/deep_res_mlp_results`.

### Reporting rule

For the presentation, the deep-MLP comparison is standardized to **static OOT Macro** metrics:

* **Primary reported metric**: OOT Macro PR-AUC on $\tau=35,\ldots,49$.
* **Secondary reported metric**: OOT Macro F1 on the same snapshots.
* Development-split scores are not used in the presentation figures because they stop before the shutdown and recovery period.

This matters here because the pre-shutdown development split can make brittle graph heads look safer than they are. The post-shutdown OOT window is the realistic stress test.

### What failed: residual heads

The residual variants were informative negative results:

* `run0`: wide residual LayerNorm + SiLU head underperformed the previous graph-MLP benchmark.
* `run1`: smaller residual LayerNorm + SiLU head still underperformed.

The likely reason is that residual capacity made it easier for the head to memorize pre-shock graph motifs. In a non-stationary transaction graph, additional head capacity is not automatically helpful.

### What worked: small LayerNorm + SiLU without residuals

The first useful improvement came from removing residual connections while keeping LayerNorm and SiLU:

* `run2`: `(128, 128)` LayerNorm + SiLU, no residual.
* `run3`: `(256, 128)` tapered LayerNorm + SiLU, no residual.

`run2` was the stronger targeted trial under OOT Macro reporting, improving the directional/PCA graph-MLP benchmark without adding residual capacity. This motivated the broader phase sweeps.

### Overview of the sweep phases

The experiment was conducted sequentially, reading each phase through OOT Macro PR-AUC and OOT Macro F1 for the presentation.

#### Phase A: Architecture depth and width

* **Scope**: swept SGC neighborhood depth $K \in \{1,2,3\}$ and MLP hidden sizes.
* **Base configuration**: PCA features, directional message passing, late topology, no residuals.
* **Finding**: $K=3$ with a compact `(64, 64)` head was best by OOT Macro PR-AUC: $0.305$.

The key lesson is that wider heads were not better. The propagated graph representation is already rich; a large head can overfit brittle pre-shock topology.

#### Phase B: Graph feature controls

* **Scope**: fixed $K=3$, `(64, 64)` and swept Base/PCA, directional/symmetric propagation, and topology injection.
* **Finding**: PCA + directional propagation + late topology remained the best OOT-Macro configuration.

This result complements the earlier SGC grid: PCA is especially useful when deeper propagation is used, because it regularizes the oversmoothed high-hop representation.

#### Phase C: Dropout regularization

* **Scope**: fixed the graph configuration and swept dropout $\in \{0.1,0.2,0.3,0.4\}$.
* **OOT Macro winner**: dropout $0.3$ with OOT Macro PR-AUC $0.305$ and OOT Macro F1 $0.262$.
* **Diagnostic note**: dropout $0.4$ looked slightly better on the development split, but it was worse on the post-shutdown OOT window and is therefore not the presentation choice.

#### Phase D: Optimizer tuning

* **Scope**: swept AdamW learning rate and weight decay around the compact LayerNorm + SiLU head.
* **OOT Macro winner within Phase D**: learning rate $0.01$, weight decay $0.0001$, with OOT Macro PR-AUC $0.297$.
* **Finding**: optimizer tuning did not beat the Phase C OOT-Macro finalist; the simpler default setting remained the better presentation result.

### Final graph-head configuration

The final OOT-Macro graph head is:

```python
use_mlp_head = True
use_multiscale_prop = True
use_layernorm = True
activation = "silu"
use_residual = False

mlp_hidden = (64, 64)
mlp_dropout = 0.3

sgc_k = 3
use_directional_prop = True
use_graph_structural = True
topo_injection_mode = "late"

use_pca = True
pca_variance = 0.98

sgc_lr = 0.01
sgc_weight_decay = 0.0005
```

### Final interpretation

The experiment should not be presented as a victory for residual MLPs. The better conclusion is:

> Multiscale SGC benefits from a small LayerNorm + SiLU classifier head, but residual connections and wider heads are counterproductive in this non-stationary graph setting.

The final graph head improves the graph-MLP OOT Macro comparison relative to many residual attempts, but it does **not** beat the strongest tabular baselines such as RandomForest and XGBoost.
