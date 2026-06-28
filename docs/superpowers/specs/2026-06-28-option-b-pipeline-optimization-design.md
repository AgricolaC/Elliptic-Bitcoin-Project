---
name: option-b-pipeline-optimization
description: Fix sweep.py data-integrity bugs + propagate explicit hyperparameter column schema downstream to parser, aggregator, and CSV-2
metadata:
  type: project
---

# Option B: Upstream Fixes & Schema Propagation

**Date:** 2026-06-28  
**Status:** Approved by user

## Context

`sweep.py` recently refactored decay ablation sweep names from `Ablation: Decay λ=0.25 on K=2, Dir=F, Topo=late (Var Base)` to `Ablation: Decay on K=2, Dir=F, Topo=late` with `Decay_Lambda=0.25` stored as an explicit CSV-1 column. This broke three downstream consumers that parse `λ` out of the sweep name string. Simultaneously, 6 data-integrity bugs in `sweep.py` corrupt results on re-run.

---

## Part 1: `sweep.py` Upstream Bug Fixes

Six bugs, each independent, each confirmed in the code.

### Bug 1 — Closure capture in Phase 5 decay loop (lines ~1511, ~1546)
**Root cause:** `_make_result_wrapped` defined inside a `for lam in [0.05, 0.25, 0.50]` loop captures `lam` by reference. All three closures see `lam=0.50` at call time.  
**Fix:** Use a default-argument capture: `def _make_result_wrapped(*args, _lam=lam, **kwargs):` and pass `decay_lambda=_lam`.

### Bug 2 — Phase 1.5 champion selection uses test metric (selection leakage)
**Root cause:** `nomp_grouped` uses `Static_OOT_Macro_F1` (OOT = test set) to rank No-MP configurations before walk-forward. This is selection leakage.  
**Fix:** Use `Static_Val_Macro_F1` for ranking. Fall back to `Static_Val_Pooled_F1` if Val_Macro is N/A.

### Bug 3 — `sweep_key` undefined in `execute_sweep()` (line ~1255) and `build_mlp_variation_specs()` (line ~362)
**Root cause:** `best_grid_key = sweep_key` and `specs.append((sweep_key, name, cfg))` both reference `sweep_key` which is never assigned in scope.  
**Fix in `execute_sweep()`:** Replace `sweep_key` with `name` (the local loop variable) when tracking `best_grid_key`.  
**Fix in `build_mlp_variation_specs()`:** Remove `sweep_key` from the returned tuple; return `(name, cfg)` pairs.  
**Fix in caller (Phase 2.5 loop, line ~1336):** Update the `for sweep_key, name, cfg in build_mlp_variation_specs(...)` unpack to `for name, cfg in build_mlp_variation_specs(...)`.

### Bug 4 — Phase 2.5 permanently gated by `elif False:`
**Root cause:** The condition `elif False:` at line ~1326 means Phase 2.5 MLP variation runs never execute.  
**Fix:** Change `elif False:` to `else:` (or `if phase25_targets:`). Phase 2.5 is already guarded by `if not phase25_targets: print(...); return`.

### Bug 5 — Decay ablation `completed_sweeps` key collision
**Root cause:** All three lambda values for a given champion use the same `w_name = f"Ablation: Decay on {base_name}"` as the completed-sweeps key. On re-run, all three hits are skipped after the first.

**Fix — two-part:**
- **Sweep name stays unique per λ in the Sweep column:** Use `w_name = f"Ablation: Decay λ={lam} on {base_name}"` so each run gets a distinct row in CSV-1 (and therefore a distinct `completed_sweeps` 3-tuple). The `Decay_Lambda` column still stores the float value for column-first downstream consumers.
- **CSV resume compatibility:** The startup CSV loading (which builds `completed_sweeps` from existing rows) naturally picks up the distinct sweep names, so previously-completed runs are still skipped correctly.

This is the only fix that simultaneously disambiguates the completed_sweeps key AND restores the `_LAM_RE` regex fallback in `sweep_parser.py` (since `λ=X` is back in the name). Note: this slightly reverts the recent refactor decision of removing λ from the name, but it is the correct tradeoff: the sweep name is the primary key and must be unique per configuration.

### Bug 6 — CSV-2 always appends (no dedup guard)
**Root cause:** `_write_csv2()` in `ablation_validation.py` unconditionally `pd.concat`s new rows onto the existing CSV. Re-runs produce duplicate `(Sweep, Seed, Tau)` triples.  
**Fix:** After `pd.concat`, drop duplicates on `["Sweep", "Seed", "Tau"]`, keeping the last (most recent) entry.

---

## Part 2: Schema Propagation — `aggregate_sweeps()` in `temporal_analysis.py`

**Problem:** `aggregate_sweeps()` groups by `["Base_Sweep", "Variation"]` and aggregates only `numeric_cols`. The hyperparameter columns (`SGC_K`, `Multiscale_Prop`, `Directionality`, `Topological_Injection`, `Decay_Lambda`, `Feature_Set`, `Threshold_Method`) are silently dropped from `final_aggregated_results.csv`.

**Fix:** After the numeric `.agg(['mean', 'std'])`, join back a `.first()` of the hyperparameter columns so they appear in the aggregated output.

```python
HYPER_COLS = ["Feature_Set", "SGC_K", "Multiscale_Prop", "Directionality",
              "Topological_Injection", "Decay_Lambda", "Threshold_Method"]

# Existing:  agg_df = df.groupby(...)[numeric_cols].agg(...)
# Add:
hyper_df = df.groupby(["Base_Sweep", "Variation"])[HYPER_COLS].first().reset_index()
agg_df = agg_df.merge(hyper_df, on=["Base_Sweep", "Variation"], how="left")
```

---

## Part 3: `sweep_parser.py` — Column-First Reading

**Problem:** `add_parsed_columns()` calls `parse_sweep()` on every row, which relies on `λ=X` being present in the sweep name string. New decay rows have no `λ` in the name → `lam` is always `None` → `select(df, lam=0.25)` returns 0 rows.

**Fix:** Add an optional `df` parameter to `add_parsed_columns()`. When the dataframe has explicit hyperparameter columns, override the parsed fields:

```python
EXPLICIT_COL_MAP = {
    "lam":  "Decay_Lambda",
    "K":    "SGC_K",
    "Dir":  "Directionality",    # bool in CSV; coerce to bool
    "Topo": "Topological_Injection",
}

def add_parsed_columns(df, sweep_col="Sweep", prefix="_"):
    out = df.copy()
    info = df[sweep_col].map(parse_sweep)
    for field in ("family", "family_tag", "lam", "K", "Dir", "Topo", "seed", "variation"):
        out[f"{prefix}{field}"] = info.map(lambda i, f=field: getattr(i, f))

    # Column-first override
    for field, col in EXPLICIT_COL_MAP.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce") if field in ("lam", "K") else df[col]
            mask = vals.notna() & (vals != "") & (vals != "N/A")
            out.loc[mask, f"{prefix}{field}"] = vals[mask]

    return out
```

`select()` is unchanged — it reads the `_lam`, `_K` etc. columns set by `add_parsed_columns()`, which are now correct.

---

## Part 4: Backfill CSV-2 Hyperparameter Columns

**Problem:** `walk_forward_timesteps.csv` has no hyperparameter columns — notebooks cannot filter timesteps by `lam`, `K`, etc. without a manual join.

**Fix (two parts):**

**4a. `_write_csv2()` in `ablation_validation.py`:** Accept an optional `hyper_cols: dict` kwarg and stamp each row with those values before writing.

**4b. One-time backfill:** Extend (or replace) the root-level `backfill_csv.py` to join `walk_forward_timesteps.csv` against `sweep_results.csv` on `["Sweep", "Seed"]` and write the hyperparameter columns. Columns to carry: `["SGC_K", "Multiscale_Prop", "Directionality", "Topological_Injection", "Decay_Lambda", "Feature_Set", "Variation"]`.

---

## File Change Map

| File | Change |
|------|--------|
| `source/sweep.py` | Bug 1 (closure), 2 (leakage), 3 (sweep_key), 4 (elif False), 5 (completed_sweeps key) |
| `source/evaluation/ablation_validation.py` | Bug 6 (CSV-2 dedup); Part 4a (stamp hyper cols in `_write_csv2`) |
| `source/analysis/temporal_analysis.py` | Part 2 (carry hyper cols in `aggregate_sweeps`) |
| `source/reporting/sweep_parser.py` | Part 3 (column-first override in `add_parsed_columns`) |
| `backfill_csv.py` (root) | Part 4b (one-time CSV-2 backfill) |

---

## What Is NOT in Scope

- `dm` caching / eliminating duplicate `EllipticDataModule.setup()` calls (noted as a performance opportunity, deferred)
- Removing `sweep_fast.py` (deferred)
- Purging the 15 `patch_*.py` files at root (deferred)
- Modifying test suite (no new correctness-changing behavior visible to tests)

---

## Verification Steps

1. **Bug 1 check:** After fix, run Phase 5 once, verify `Decay_Lambda` column in new CSV-1 rows contains all three of 0.05, 0.25, 0.50 (not just 0.50).
2. **Bug 2 check:** Verify Phase 1.5 champion selection uses `Static_Val_Macro_F1` in code.
3. **Bug 3 check:** Verify `execute_sweep()` uses `name` not `sweep_key` for `best_grid_key`; verify `build_mlp_variation_specs` returns `(name, cfg)` pairs.
4. **Bug 4 check:** Phase 2.5 target-expansion code runs when eligible targets exist.
5. **Bug 5 check:** Re-run produces three distinct completed_sweeps keys per champion (one per lambda value).
6. **Bug 6 check:** Re-run of any ablation does not grow `walk_forward_timesteps.csv` row count.
7. **Aggregator check:** `python source/analysis/temporal_analysis.py --action aggregate` → `final_aggregated_results.csv` contains `SGC_K`, `Decay_Lambda` columns.
8. **Parser check:** `select(df, lam=0.25)` on a loaded `sweep_results.csv` returns non-empty rows.
9. **End-to-end:** `python source/reporting/build_notebook.py` completes without errors.
