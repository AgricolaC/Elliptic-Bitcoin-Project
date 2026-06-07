# Narrative Specification: Per-Step F1 Chart

This document specifies the exact narrative points to hit when presenting the walk-forward F1 chart during the 12-minute oral defense.

## The Visual
**File:** `walk_forward_drift_Sweep_4__Topology_Features.png` (or similar depending on ablation name)
**Key Elements:** 
- White background for projector readability.
- The red line showing F1 score across the 49 subgraphs.
- The bold, black vertical line at $t=43$ labeled "dark market shutdown".

## The 2-Minute Narrative Arc
1. **The Train/Test Gap:** Point out the $\sim 0.27$ gap between the static OOT performance ($\sim 0.90$) and the walk-forward mean ($\sim 0.63$). 
2. **Concept Drift (`nb2_timeseries_dl`):** Explicitly name this phenomenon. This isn't a failure of model capacity; this is "evolving normality" or "concept drift" inherent to the time-variant data. The underlying manifold is shifting.
3. **The Shutdown Shock:** Direct attention to $t=43$. The catastrophic drop in F1 is the structural break caused by the shutdown of a major darknet market. This perfectly illustrates why standard i.i.d. assumptions fail on this dataset.
4. **Conclusion:** Our expanding-window protocol is the rigorous, honest way to measure performance because it respects the strict causal boundary ($max(train) < \tau$).
