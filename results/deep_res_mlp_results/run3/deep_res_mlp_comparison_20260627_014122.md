# LayerNorm + SiLU Tapered MLP Results

Generated: 2026-06-27T01:42:42

Worker caps used: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB=1, torch threads=1.

Head: LayerNorm + SiLU + no residual + mlp_hidden=(256, 128).

Benchmark: `Grid: K=2, Dir=T, Topo=late (Seed 44, Var PCA)`

| Metric | Previous MLP | Run2 (128,128) | Run3 (256,128) | Δ Run3 vs Previous | Δ Run3 vs Run2 |
|---|---:|---:|---:|---:|---:|
| Val_Pooled_PRAUC | 0.949000 | 0.946772 | 0.938895 | -0.010105 | -0.007877 |
| Val_Macro_PRAUC | 0.867000 | 0.875940 | 0.869662 | +0.002662 | -0.006278 |
| OOT_Pooled_PRAUC | 0.350000 | 0.425243 | 0.415618 | +0.065618 | -0.009625 |
| OOT_Macro_PRAUC | 0.265000 | 0.282710 | 0.280031 | +0.015031 | -0.002679 |
| Val_Pooled_F1 | 0.871000 | 0.865184 | 0.857414 | -0.013586 | -0.007770 |
| Val_Macro_F1 | 0.804000 | 0.822489 | 0.812086 | +0.008086 | -0.010403 |
| OOT_Pooled_F1 | 0.394000 | 0.460481 | 0.471032 | +0.077032 | +0.010551 |
| OOT_Macro_F1 | 0.206000 | 0.265999 | 0.266427 | +0.060427 | +0.000428 |
