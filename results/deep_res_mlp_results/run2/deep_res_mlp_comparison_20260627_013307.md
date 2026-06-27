# LayerNorm + SiLU Small MLP A/B Results

Generated: 2026-06-27T01:34:54

Worker caps used: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB=1, torch threads=1.

Head: LayerNorm + SiLU + no residual + mlp_hidden=(128, 128).

## val_macro_champion

Benchmark: `Grid: K=1, Dir=F, Topo=late (Seed 44, Var Base)`

| Metric | Previous MLP | LN+SiLU No Residual | Δ |
|---|---:|---:|---:|
| Val_Pooled_PRAUC | 0.943000 | 0.943137 | +0.000137 |
| Val_Macro_PRAUC | 0.892000 | 0.868067 | -0.023933 |
| OOT_Pooled_PRAUC | 0.337000 | 0.242329 | -0.094671 |
| OOT_Macro_PRAUC | 0.255000 | 0.214011 | -0.040989 |
| Val_Pooled_F1 | 0.885000 | 0.854639 | -0.030361 |
| Val_Macro_F1 | 0.832000 | 0.798256 | -0.033744 |
| OOT_Pooled_F1 | 0.467000 | 0.198582 | -0.268418 |
| OOT_Macro_F1 | 0.269000 | 0.124457 | -0.144543 |

## val_pooled_champion

Benchmark: `Grid: K=2, Dir=T, Topo=late (Seed 44, Var PCA)`

| Metric | Previous MLP | LN+SiLU No Residual | Δ |
|---|---:|---:|---:|
| Val_Pooled_PRAUC | 0.949000 | 0.946772 | -0.002228 |
| Val_Macro_PRAUC | 0.867000 | 0.875940 | +0.008940 |
| OOT_Pooled_PRAUC | 0.350000 | 0.425243 | +0.075243 |
| OOT_Macro_PRAUC | 0.265000 | 0.282710 | +0.017710 |
| Val_Pooled_F1 | 0.871000 | 0.865184 | -0.005816 |
| Val_Macro_F1 | 0.804000 | 0.822489 | +0.018489 |
| OOT_Pooled_F1 | 0.394000 | 0.460481 | +0.066481 |
| OOT_Macro_F1 | 0.206000 | 0.265999 | +0.059999 |
