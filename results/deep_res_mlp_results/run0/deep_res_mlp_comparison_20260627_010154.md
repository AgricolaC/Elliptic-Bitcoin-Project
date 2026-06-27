# Deep Residual MLP A/B Results

Generated: 2026-06-27T01:10:17

Worker caps used: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB=1, torch threads=1.

Deep head: LayerNorm + SiLU + residual projection + mlp_hidden=(512, 256, 128).

## val_macro_champion

Benchmark: `Grid: K=1, Dir=F, Topo=late (Seed 44, Var Base)`

| Metric | Previous MLP | Deep Residual | Δ |
|---|---:|---:|---:|
| Val_Pooled_PRAUC | 0.943000 | 0.879695 | -0.063305 |
| Val_Macro_PRAUC | 0.892000 | 0.816661 | -0.075339 |
| OOT_Pooled_PRAUC | 0.337000 | 0.197478 | -0.139522 |
| OOT_Macro_PRAUC | 0.255000 | 0.175298 | -0.079702 |
| Val_Pooled_F1 | 0.885000 | 0.839980 | -0.045020 |
| Val_Macro_F1 | 0.832000 | 0.793028 | -0.038972 |
| OOT_Pooled_F1 | 0.467000 | 0.235294 | -0.231706 |
| OOT_Macro_F1 | 0.269000 | 0.132458 | -0.136542 |

## val_pooled_champion

Benchmark: `Grid: K=2, Dir=T, Topo=late (Seed 44, Var PCA)`

| Metric | Previous MLP | Deep Residual | Δ |
|---|---:|---:|---:|
| Val_Pooled_PRAUC | 0.949000 | 0.919305 | -0.029695 |
| Val_Macro_PRAUC | 0.867000 | 0.848955 | -0.018045 |
| OOT_Pooled_PRAUC | 0.350000 | 0.298273 | -0.051727 |
| OOT_Macro_PRAUC | 0.265000 | 0.256660 | -0.008340 |
| Val_Pooled_F1 | 0.871000 | 0.865486 | -0.005514 |
| Val_Macro_F1 | 0.804000 | 0.813609 | +0.009609 |
| OOT_Pooled_F1 | 0.394000 | 0.379524 | -0.014476 |
| OOT_Macro_F1 | 0.206000 | 0.217042 | +0.011042 |
