# Presentation Sweep Name Fix — Design Spec

**Goal:** Fix all stale sweep name references in `source/reporting/build_presentation.py` so the regenerated `presentation.ipynb` renders correct charts from current result CSVs.

**Out of scope:** Narrative `[PENDING PIPELINE RE-RUN]` markdown blocks — these are left for manual authoring.

---

## Background

`build_presentation.py` was written against an older naming convention (`'F1: ...'`, `'F3: ...'`, `'F4: ...'`). The pipeline was later refactored and all sweep names changed. The result CSVs (`final_aggregated_results.csv`, `final_aggregated_timesteps.csv`) now use a different convention (`'Grid: K=N, Dir=X, Topo=Y'`, `'Ablation: Decay λ=Z on ...'`, `'Baseline: ...'`, `'Best WF: ...'`). Charts that call `get_scalar()` or filter `steps` with a stale name silently return 0 or an empty series.

Additionally, the K=5 configuration was never run. The K-depth reversal slide hardcodes K=2 vs K=5 static F1 values and references a non-existent K=5 WF sweep.

---

## Task B — Verification Script

**File:** `source/reporting/check_sweep_names.py` (new, temporary diagnostic)

**What it does:**
1. Loads `results/final_aggregated_results.csv` and `results/final_aggregated_timesteps.csv`.
2. Extracts every sweep name string literal referenced in `source/reporting/build_presentation.py` via a static scan (regex for quoted strings passed to `get_scalar()` and used as filter values against `steps`).
3. Prints a hit/miss table: name → Found / MISSING.
4. For each MISSING name, prints the closest match from the CSVs (substring match).
5. For the SGC decay sweeps (`Ablation: Decay λ=X on * Base`), prints all three candidates per λ value (K=1 late, K=3 early, K=3 late) with their `WF_Pooled_F1_mean` from `final_aggregated_results.csv` so the best can be chosen per λ.

**Output:** Terminal only. No files written, no builder modified.

**Success criterion:** Every referenced name is either confirmed present or has an unambiguous replacement identified. The SGC decay candidate table shows clearly which config per λ has the highest `WF_Pooled_F1`.

---

## Task A — Targeted Fixes in `build_presentation.py`

Using Task B's output as the source of truth, edit `source/reporting/build_presentation.py` at these locations:

### 1. Baseline slide (Slide 6)

Replace the four stale baseline sweep names in the `inst` list:

| Old name | New name |
|---|---|
| `'Diagnostic: sklearn LR'` | `'Baseline: Logistic Regression (166)'` |
| `'F3d: GCN reference [2-layer]'` | `'Baseline: PyG GCN (2-layer)'` |
| `'F3a: Base XGBoost (clean)'` | TBD — confirmed by Task B |
| `'F3b: Random Forest (clean)'` | `'Baseline: RandomForest (166)'` |

### 2. K-depth reversal slide (Slide 12)

Replace K=2 vs K=5 with K=1 vs K=3:
- `labels` → `['K=1 (Shallow SGC)', 'K=3 (Deep SGC)']`
- `static_f1` → two `get_scalar()` calls pulling `Static_OOT_Pooled_F1` for the best-performing K=1 and K=3 grid configs (highest mean across `Dir` and `Topo` variants) from `final_aggregated_results.csv`
- `wf_pooled` and `wf_macro` → `get_scalar()` calls using `Best WF: Grid: K=1, Dir=F, Topo=late` and `Best WF: Grid: K=3, Dir=F, Topo=late` (or whichever Task B confirms as highest WF)

### 3. SGC decay per-timestep slide (Slide 16)

Replace:
- `'F1: SGC+MLP WF K=2 [Dir=F; Topo=None]'` → `'Best WF: Grid: K=2, Dir=F, Topo=None'`
- `'F4: SGC+MLP decay λ=0.05'` → winning candidate from Task B for λ=0.05
- `'F4: SGC+MLP decay λ=0.25'` → winning candidate from Task B for λ=0.25
- `'F4: SGC+MLP decay λ=0.5'` → winning candidate from Task B for λ=0.5

### 4. Ultimate finding slide (Slide 17)

Replace:
- `'F1: SGC+MLP WF K=2 [Dir=F; Topo=None]'` → `'Best WF: Grid: K=2, Dir=F, Topo=None'`
- `'F4: SGC+MLP decay λ=0.05'` → same winning candidate used in Slide 16 for λ=0.05

### 5. Regenerate notebook

Run `python source/reporting/build_presentation.py` and confirm it exits cleanly.

---

## Success Criteria

- `check_sweep_names.py` prints no MISSING names after Task A fixes are applied (can re-run it as a smoke test).
- `presentation.ipynb` regenerates without error.
- All charts that previously showed zero bars / empty lines now show real data.
- `[PENDING]` markdown blocks remain untouched.
