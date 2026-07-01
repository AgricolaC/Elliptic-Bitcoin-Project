## Final Conclusions & Takeaways

Across our exploratory data analysis, graph falsification diagnostics, baseline benchmarking, and deep hyperparameter sweeps, a clear and cohesive narrative emerges regarding the Elliptic Bitcoin dataset and the $\tau=43$ AlphaBay anomaly.

### 1. The $\tau=43$ Myth Busted: Prior Shift, not Representational Collapse
Our initial hypothesis framed the catastrophic model failure at $\tau=43$ to a fundamental collapse in the geometric or topological feature space. Our diagnostic falsification analysis definitively disproves this. The local node features remain highly separable across permutations, and covariate drift is surprisingly stable at the exact moment of the shock. The true cause of the anomaly is a massive **class-prior collapse**: the sheer volume of known illicit nodes drops by an order of magnitude.

### 2. The Graph Memory Trap
Deep, static Graph Neural Networks (and high-hop SGC variants) fall into a memory trap. In the stable pre-shock era ($\tau \le 42$), these models memorize deep, directed micro-motifs. When the regime shifts post-shock, the transaction network topology inevitably evolves. Models that hyper-fit to the deep, historical geometry fail to generalize out-of-time, resulting in catastrophic F1 degradation during the recovery phase.

### 3. The Power of Tabular Robustness
Complex, differentiable message-passing is slow and often counterproductive across temporal shocks. Standard tree-based tabular models (XGBoost and RandomForest) train up to 60x faster than standard PyG GCNs and completely dominate the static Out-of-Time evaluation. *Why* they dominate is worth stating carefully. A natural hypothesis is **feature selection**: of the 166 features, 72 are graph-neighborhood aggregates and 94 are purely *local*, so a decision tree can down-weight the aggregates once the post-shock neighborhood diverges, whereas a GCN blends local and neighborhood features at every layer and cannot opt out. We present this as a **hypothesis, not a measured result** — the committed experiments include no feature-importance analysis or 94-vs-166 ablation, and at the shock step itself ($\tau=43$) XGBoost also collapses to $0.000$ Macro-F1, so its advantage is *not* shock-survival. What the data *does* show is that tabular models **recover faster** after the shock ($0.393$ vs $\sim0.175$ Recovery Macro-F1) and hold a higher pre-shock ceiling.

### 4. The Winning Strategy: XGBoost + Decaying Topological Features
Standard complex message-passing is incredibly slow and often counterproductive across temporal shocks. Standard tree-based tabular models (XGBoost) completely dominate the evaluation, peaking at **$0.674$ Macro F1** when augmented with exponentially decaying multiscale graph features ($\lambda=0.5$).
If Graph Neural Networks must be utilized, deep and static is the wrong approach. Our current strategy pairs **Walk-Forward (WF) training** with **Exponential Decay**.
* **Walk-Forward training** continuously recalibrates the model and absorbs micro-shifts in topology.
* **Exponential Decay ($\lambda=0.25$)** acts as an explicit regularizer against structural overfitting. Strikingly, it allows highly complex graph architectures (e.g., `SGC K=2, Directed, Late Topology`) to overcome their previous brittleness. By explicitly forcing the model to "forget" the pre-shock AlphaBay topology, this complex directed model surges to **$0.527$ Macro F1**, becoming the champion among pure graph architectures we tested.

### 5. Limitations of This Study

We state the following limitations so that our claims are not over-read:

* **The mechanism behind the tabular advantage is unproven.** Our "feature selection" explanation for why trees beat GNNs is a hypothesis: we did not run a feature-importance analysis or a 94-local-vs-166-all ablation, so we cannot confirm that trees specifically discard the 72 aggregate features. The verified claim is only the empirical outcome (faster post-shock recovery, higher pre-shock ceiling).
* **Broadcast bias is ruled out only at one hop.** The label-separability test that falsifies broadcast bias uses a single round of propagation ($K=1$). Deeper, multi-layer message passing could still smear representations; we did not test the separability of deeper embeddings.
* **The shock step is statistically thin.** At $\tau=43$ only ~24 labelled illicit nodes exist. Separability tests there are flagged low-confidence, and Macro-F1 requires threshold calibration on a handful of positives, so shock-step metrics are fragile.
* **Limited seeds and no significance testing.** Most grid configurations were run with 2–3 seeds (some single-seed), and comparisons are reported without error bars or significance tests. Small F1 gaps (on the order of $\sim0.01$) should not be treated as reliable rankings.
* **Evaluation ignores the unlabelled majority.** Roughly three-quarters of nodes are `unknown` and are excluded from loss and metrics, yet they participate in neighborhood features and message passing. Our reported performance therefore characterizes only the labelled subgraph; true operational performance on unlabelled traffic is unmeasured.
* **The subpopulation-shift claim is inferential.** We argue the ~24 shock survivors are a non-random subsample of the pre-shock illicit population, but we did not directly test whether they occupy a distinct feature or topology region.
* **Discrete-snapshot scope.** All models operate on discrete snapshots; the continuous-time and evolving-parameter approaches discussed below were *proposed but not implemented*, so our conclusions about graph-model brittleness are specific to the static-snapshot regime.
* **Drift metrics are unconditional and low-dimensional.** Covariate drift is measured over all nodes (not class-conditional) and, for the Wasserstein component, on a 3-component PCA projection; these summarize whole-population shift and are not sensitive to illicit-specific migration.

### 6. Future Research Directions: Beyond Discrete Snapshots
Our findings expose the brittleness of discrete, static graph modeling when faced with financial regime shifts. Drawing on State-of-the-Art methodologies, future work should explore:
* **Native Temporal Graphs (TGNs)**: Rather than treating time as discrete snapshots ($G_1, \dots, G_{49}$), models like **Temporal Graph Networks (TGN)** maintain continuous node memory that updates with every single transaction edge, naturally absorbing micro-shifts without needing a clunky Walk-Forward wrapper.
* **Evolving Parameters (EvolveGCN)**: Using RNNs to evolve the GCN weight matrices themselves across timesteps allows the network to adapt its propagation logic to new topological regimes dynamically.
* **Heterophilic Message Passing**: Illicit actors often attempt to mask their funds by routing them through licit hubs (exchanges). Standard homophilic aggregation blurs this signal. Advanced message-passing schemes tailored for **Heterophily** could preserve the sharp contrast between a fraudster and the legitimate exchange they cash out through.

Beyond new architectures, several directions follow directly from the failure mode and the limitations we identified:

* **Imbalance-first learning at the head**: Because the $\tau=43$ failure is head-level class imbalance rather than representational collapse, the most direct lever is the classifier head — **focal / class-balanced loss**, minority oversampling or synthetic augmentation, and **calibrated or conformal decision thresholds** tuned for extreme low-prevalence regimes.
* **Testing the feature-selection hypothesis directly**: Run **feature-importance analysis** (e.g. SHAP) on the tree models and a controlled **94-local-only vs 166-all-features ablation** across the temporal split, to confirm or refute that trees survive by down-weighting the 72 aggregate features — closing the largest open question in our tabular-vs-graph comparison.
* **Exploiting the `Unknown` class**: **Semi-supervised** or **positive-unlabelled (PU)** learning could turn the large unlabelled majority from passive noise into usable signal, rather than masking it out of loss and metrics.
* **Depth-resolved broadcast-bias analysis**: Extend the raw-vs-propagated separability test to **deeper ($K>1$, multi-layer) embeddings** to determine whether representational smearing emerges at depth, closing the single-hop caveat behind our broadcast-bias falsification.
* **Prior-shift-aware detection**: Methods explicitly designed for **label-prior shift** — density-ratio / importance-weighted estimation, or anomaly scores calibrated to a time-varying base rate — target the actual failure mode we diagnosed at $\tau=43$.
* **Uncertainty quantification under scarcity**: **Conformal prediction** or Bayesian heads could supply reliable confidence at the shock step, where point predictions on ~24 positives are untrustworthy.

## References
This investigation builds upon and benchmarks against several foundational works in Geometric Deep Learning and the original Elliptic dataset publication:
* **The Elliptic Dataset:** Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*
* **GCN Baseline:** Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*
* **SGC Baseline:** Wu et al., *Simplifying Graph Convolutional Networks*
* **Phase Transition Diagnostics:** *Persistent Homology analysis of Phase Transitions* (Considered for $\tau=43$ geometric diagnostics, but bypassed after simple topological permutation tests falsified the representational collapse thesis).
* **Dynamic/Temporal Models (Future Work):**
  * Pareja et al., *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs*
  * Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs*
  * Hamilton et al., *Inductive Representation Learning on Large Graphs (GraphSAGE)*
