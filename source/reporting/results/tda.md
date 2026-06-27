---
TDA Module: What the Results Tell You

1. topological_diagnostics.csv — Cross-timestep geometry (tda_diagnostic.py:
What it measures: Consecutive-pair MMD² (Maximum Mean Discrepancy) between pclouds at τ and τ+1. Two clo
- Base (Cloud A): all nodes, capped to M=800, same M across all τ           - ClassCond_Illicit (Cloud B(the minimum qualifyingsnapshot), σ fixed globally                                                 
Base (all nodes) findings:                                                  All 48 transitions have very28, median ≈ 0.008). τ=42→43is ranked 10th with median MMD²=0.0143. The global node feature cloud barelyat τ=43. This is the geometr Bitcoin transaction graph'sraw feature distribution is stationary; there's no manifold shift at the coltimestep.
                                                                            ClassCond_Illicit (illicit oing topology:
The illicit subpopulation is much less geometrically stable than the full grtransitions by median MMD²:
                                                                            ┌────────────┬─────────────┐
│ Transition │ Median MMD² │                                                ├────────────┼─────────────┤
│ τ=24→25    │ 0.152       │                                                ├────────────┼─────────────┤
│ τ=28→29    │ 0.146       │                                                ├────────────┼─────────────┤
│ τ=29→30    │ 0.145       │                                                ├────────────┼─────────────┤
│ τ=42→43    │ 0.108       │                                                ├────────────┼─────────────┤
│ τ=23→24    │ 0.082       │                                                └────────────┴─────────────┘
                                                                            The illicit subpopulation unc reorganizations (τ≈23–25 and τ≈28–30) that are larger than the τ=43 collapse. These don't cause F1 failurprevalence is adequate. τ=42's 4th — the "shock" isprimarily a prevalence event (239→24 illicit nodes), not a unique geometric singularity.

After the collapse, τ=43→44 entially zero, negativeunbiased estimator from very small M). The 24 surviving illicit nodes look unremarkably similar to the s.

---
2. label_separability.csv — Within-timestep class geometry (label_separability.py:176)

What it measures: Size-matched MMD² permutation test between illicit and licit node clouds at each τ, for two representations:
- Raw: StandardScaler-normalized raw features
- Prop_k1_Dir0: one-hop symmetric SGC propagation S̃X (matching the model's operator)

The operator is built in label_separability.py:75 as D⁻¹/²(max(A,Aᵀ)+I)D⁻¹/² — thisexactly mirrors gcn_norm in layers.py, which is important: you're testing the actual representation the model uses, not a surrogate.

Coverage: Raw separable at 44/49 τ, Prop at 45/49 τ. Most failures are early timesteps (τ=1–6) with only 5–17 illicit nodes — too few for statistical power.

τ=42 vs τ=43 (the key comparison):

┌─────┬───────────┬────────────┬────────────┬──────────────────────────┐
│  τ  │ n_illicit │  Raw sep   │  Prop sep  │          Notes           │
├─────┼───────────┼────────────┼────────────┼──────────────────────────┤
│ 42  │ 239       │ 100% seeds │ 100% seeds │ High-power reference     │
├─────┼───────────┼────────────┼────────────┼──────────────────────────┤
│ 43  │ 24        │ 80% seeds  │ 100% seeds │ Only 2/10 seeds fail raw │
└─────┴───────────┴────────────┴────────────┴──────────────────────────┘

This is the decisive finding: SGC propagation at τ=43 is not collapsing the illicit representation. The propagated space is more robustly separable than raw at τ=43 (100% vs 80% seed agreement). The two seeds that fail raw (seed=3: p=0.063, seed=9: p=0.065) are borderline and have the smallest MMD² values — consistent with randomsampling variance at n=24.

---
3. falsification_log.csv — The verdict (tda_diagnostic.py:443,label_separability.py:256)

Six rows, summarized:

1. Base MMD Z = 0.054 → SILENT (expected; explained above)
2. ClassCond_Illicit MMD Z = 6.17, seed_frac=0.8 → SILENT because rank-1 criterionfails (τ=42 is rank-4 among illicit transitions, behind τ=24, 28, 29)
3. ClassCond_Illicit permtest p=0.0 → SIGNIFICANT (the illicit cloud does change atτ=43, but it's a prevalence effect)
4. Decision matrix: A✗B✗ (World γ) — τ=43 is a label-prevalence event, not a geometric transition
5. H3 (Wasserstein null) → DEFERRED (needs ripser install, not needed given World γ)
6. Broadcast-bias 2×2: NOT_BROADCAST_BIAS — both raw and prop separable at τ=43; soften to "head-level imbalance"

---
Topological Insight Summary

The most presentation-worthy insights, ordered by novelty:

1. The illicit subpopulation is geometrically restless. Three structural reorganizations at τ≈23–25 and τ≈28–30 (Cloud B MMD² ~0.15) are larger than τ=43. This suggests Bitcoin illicit activity clusters shift their transaction graph neighborhood structure multiple times, independent of labeling scarcity. These could correspond to real-world events (exchange collapses, enforcement actions in the original dataset timeline).

2. τ=43 is a prevalence event, not a manifold event. The global feature geometry isflat (Cloud A SILENT), and the illicit feature geometry survives propagation (100% seed agreement in Prop space). The model fails because it receives 24 positivetraining-adjacent examples, not because the representation is corrupted.

3. SGC propagation does not amplify the imbalance geometrically. At τ=43, propagation actually tightens separability (80%→100% seeds). The smooth averaging of S̃ helpsconcentrate the 24 illicit nodes relative to their licit neighbors, rather than washing them out. The failure is entirely upstream in the classifier head (thresholdcalibration with <10 positives triggers the local-quantile fallback).

4. The representation is stable across the test period. 44–45 out of 49 τ show significant illicit/licit separation in both raw and propagated space. Theclassifier's struggle is concentrated at specific low-prevalence timesteps, not at a broad representational breakdown.