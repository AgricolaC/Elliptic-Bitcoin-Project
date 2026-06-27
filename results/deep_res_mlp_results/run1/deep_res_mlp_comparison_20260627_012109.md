# Deep Residual Small MLP A/B Results

Generated: 2026-06-27T01:23:12

Worker caps used: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB=1, torch threads=1.

Deep head: LayerNorm + SiLU + residual projection + mlp_hidden=(128, 128).

## val_macro_champion

Benchmark: `Grid: K=1, Dir=F, Topo=late (Seed 44, Var Base)`

| Metric | Previous MLP | Small Residual | Δ |
|---|---:|---:|---:|
| Val_Pooled_PRAUC | 0.943000 | 0.858103 | -0.084897 |
| Val_Macro_PRAUC | 0.892000 | 0.836204 | -0.055796 |
| OOT_Pooled_PRAUC | 0.337000 | 0.201901 | -0.135099 |
| OOT_Macro_PRAUC | 0.255000 | 0.210676 | -0.044324 |
| Val_Pooled_F1 | 0.885000 | 0.883412 | -0.001588 |
| Val_Macro_F1 | 0.832000 | 0.840563 | +0.008563 |
| OOT_Pooled_F1 | 0.467000 | 0.356535 | -0.110465 |
| OOT_Macro_F1 | 0.269000 | 0.214389 | -0.054611 |

## val_pooled_champion

Benchmark: `Grid: K=2, Dir=T, Topo=late (Seed 44, Var PCA)`

| Metric | Previous MLP | Small Residual | Δ |
|---|---:|---:|---:|
| Val_Pooled_PRAUC | 0.949000 | 0.869245 | -0.079755 |
| Val_Macro_PRAUC | 0.867000 | 0.836870 | -0.030130 |
| OOT_Pooled_PRAUC | 0.350000 | 0.323686 | -0.026314 |
| OOT_Macro_PRAUC | 0.265000 | 0.252370 | -0.012630 |
| Val_Pooled_F1 | 0.871000 | 0.869352 | -0.001648 |
| Val_Macro_F1 | 0.804000 | 0.795497 | -0.008503 |
| OOT_Pooled_F1 | 0.394000 | 0.365488 | -0.028512 |
| OOT_Macro_F1 | 0.206000 | 0.201580 | -0.004420 |
