# Phase C Dropout Static Sweep

Generated: 2026-06-27T15:17:15

Scope: fixed K=3, hidden=(64,64), PCA + directional + topo=late, LayerNorm + SiLU + no residual. Varies dropout.

Completed runs: 12 / 12

## Winners

| Selection | Dropout | Val Macro PR-AUC | Val Pooled PR-AUC | OOT Macro PR-AUC | OOT Pooled PR-AUC |
|---|---:|---:|---:|---:|---:|
| Validation Macro PR-AUC | 0.4 | 0.880913 | 0.935630 | 0.289828 | 0.451959 |
| Validation Pooled PR-AUC | 0.2 | 0.856512 | 0.935924 | 0.267382 | 0.401610 |
| Oracle OOT Pooled PR-AUC | 0.3 | 0.880434 | 0.934552 | 0.304869 | 0.467450 |

## Configurations by validation Macro PR-AUC

| Rank | Dropout | Val Macro PR-AUC mean±std | Val Pooled PR-AUC mean±std | OOT Macro PR-AUC mean±std | OOT Pooled PR-AUC mean±std |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4 | 0.880913±0.011418 | 0.935630±0.015024 | 0.289828±0.014383 | 0.451959±0.039146 |
| 2 | 0.3 | 0.880434±0.011822 | 0.934552±0.011538 | 0.304869±0.034290 | 0.467450±0.049436 |
| 3 | 0.2 | 0.856512±0.039105 | 0.935924±0.012557 | 0.267382±0.086700 | 0.401610±0.143974 |
| 4 | 0.1 | 0.855622±0.013163 | 0.920053±0.006141 | 0.284861±0.020988 | 0.407529±0.042821 |
