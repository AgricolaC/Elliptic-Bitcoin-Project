# Phase B Graph-Control Static Sweep

Generated: 2026-06-27T15:06:30

Scope: fixed K=3, hidden=(64,64), LayerNorm + SiLU + no residual. Varies PCA/Base, directional/symmetric, and topology late/early/none.

Completed runs: 36 / 36

## Winners

| Selection | Variation | Directional | Topology | Val Macro PR-AUC | Val Pooled PR-AUC | OOT Macro PR-AUC | OOT Pooled PR-AUC |
|---|---|---:|---|---:|---:|---:|---:|
| Validation Macro PR-AUC | PCA | True | late | 0.880434 | 0.934552 | 0.304869 | 0.467450 |
| Validation Pooled PR-AUC | Base | True | late | 0.842567 | 0.938798 | 0.212727 | 0.310450 |
| Oracle OOT Pooled PR-AUC | PCA | True | late | 0.880434 | 0.934552 | 0.304869 | 0.467450 |

## Top configurations by validation Macro PR-AUC

| Rank | Variation | Directional | Topology | Val Macro PR-AUC mean±std | Val Pooled PR-AUC mean±std | OOT Macro PR-AUC mean±std | OOT Pooled PR-AUC mean±std |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | PCA | True | late | 0.880434±0.011822 | 0.934552±0.011538 | 0.304869±0.034290 | 0.467450±0.049436 |
| 2 | Base | False | none | 0.868586±0.022679 | 0.935724±0.003022 | 0.259954±0.079212 | 0.370761±0.156011 |
| 3 | PCA | True | none | 0.863363±0.011331 | 0.912135±0.024266 | 0.281831±0.039489 | 0.431373±0.056293 |
| 4 | Base | False | late | 0.860558±0.002299 | 0.930160±0.010433 | 0.265305±0.043995 | 0.369332±0.071401 |
| 5 | PCA | True | early | 0.856573±0.013476 | 0.924707±0.007685 | 0.259656±0.073797 | 0.389057±0.129215 |
| 6 | Base | False | early | 0.855429±0.024830 | 0.929173±0.008179 | 0.226836±0.053917 | 0.315116±0.124346 |
| 7 | PCA | False | none | 0.851663±0.018401 | 0.905468±0.014963 | 0.261143±0.015111 | 0.378457±0.036527 |
| 8 | Base | True | early | 0.848576±0.017869 | 0.932054±0.014491 | 0.229257±0.023008 | 0.361291±0.053365 |
| 9 | Base | True | late | 0.842567±0.035029 | 0.938798±0.015223 | 0.212727±0.035916 | 0.310450±0.093797 |
| 10 | Base | True | none | 0.840028±0.018831 | 0.933127±0.015447 | 0.227536±0.046969 | 0.347220±0.090130 |
| 11 | PCA | False | early | 0.833497±0.028080 | 0.901392±0.015172 | 0.254369±0.050265 | 0.306464±0.078203 |
| 12 | PCA | False | late | 0.823834±0.021064 | 0.898365±0.018111 | 0.284417±0.068908 | 0.369181±0.100091 |
