# Presentation Notebook Builder — Design Spec

**Date:** 2026-06-27  
**Status:** Approved (user confirmed "Go ahead" after full design walkthrough)

---

## What We're Building

A local Python script (`source/reporting/build_notebook.py`) that generates
`presentation/presentation.ipynb` — a Colab-native, self-contained research presentation
notebook for the Elliptic Bitcoin project defense before examiners Vaccarino and Gasparini.

The notebook reads like a paper with executable figures: narrative prose is baked in as
static markdown cells (visible immediately on open, no execution required), and data-driven
plots are generated at Colab runtime by loading CSVs from Google Drive.

---

## Architecture: Two-Module Approach (Approach B)

```
source/reporting/
  sweep_parser.py       — Standalone string parser for Sweep column identifiers
  build_notebook.py     — Builder: reads markdown + CSVs → emits presentation.ipynb

tests/
  test_sweep_parser.py  — 50+ test cases against REAL CSV strings (not fabricated)

presentation/
  presentation.ipynb    — Generated artifact (not committed / gitignored)
```

The parser is used in two contexts:
1. **Build time** — `build_notebook.py` imports it to validate sweep references before
   emitting code cells.
2. **Colab runtime** — the builder reads `sweep_parser.py` as text and embeds it verbatim
   as an early code cell (Cell 2), so `select()` / `add_parsed_columns()` are available
   to all plot cells without a Drive dependency.

---

## Key Design Decisions

### D1: Colab-native artifact, built locally

The builder runs locally (you have the markdown files and CSVs). The output is an `.ipynb`
that you upload to Colab and "Run All." There is no local `python build_notebook.py` step
for the examiner.

### D2: Narrative embedded at build time

Markdown source from the seven `.md` files is read by the builder and written as
`cell_type: "markdown"` cells. The examiner sees rendered prose immediately on open — no
cells to run first. This is a defense artifact, not a development tool.

### D3: Corrected sweep_parser (critical — previous version was broken)

An earlier AI-generated `sweep_parser.py` was built on fabricated sweep string formats
("F1: Base XGBoost WF [v2]", "F4: SGC+MLP decay λ=0.5") that do not exist in any CSV.
The test suite passed only because it tested the same fabricated strings. Running it against
real CSV strings yielded 8/8 key filter failures.

The correct grammar (verified against all 200+ unique Sweep strings):
```
Ablation: Decay λ=X on N Dir Topo Var   (N=1/2/3, Dir=T/F, not Dir=T/F tokens)
Ablation: Decay λ=X on XGBoost
Ablation: IPCA N Dir Topo               (positional, no K=/Dir= tokens)
Baseline: Name (args)
Sweep N: SGC (baseline) [(Seed N, Var X)]
Sweep N: + MLP Head [(Seed N, Var X)]
Grid: K=N, Dir=X, Topo=Y [(Seed N, Var X)]
Best WF: Grid: K=N, Dir=X, Topo=Y ...  ("Best WF:" prefix requires unified Grid: check)
Best WF: Sweep N: ...
Ensemble: ...
```

### D4: Section scope — 6 plotted + 1 static

- §1–4, §6–7: full plot cells, PRAUC primary, F1 secondary.
- §5 (Deep Res MLP): two markdown cells only — narrative + static table read from
  `phaseX_aggregated.csv` at build time. No code cell. The incremental F1 gains from the
  phase sweep are better explained verbally than as a bar chart.

### D5: Primary metric everywhere is PRAUC

`Static_OOT_Pooled_PRAUC` (static sections) and `PRAUC` (temporal sections). F1 is always
secondary. This matches the project's evaluation protocol and handles class imbalance
correctly.

### D6: Low_Confidence shading required

Every temporal plot must shade timesteps where `Low_Confidence == True` (grey bands).
Recovery window τ=45–46 has N_illicit as low as 2; un-shaded, those points read as signal.

### D7: Seed transparency

- Static OOT sections (§3, §4): Seeds 42, 43, 44 — show mean ± std error bars.
- WF / decay sections (§6, §7): Seed 42 only — every title/caption includes
  "n=1 seed, Seed=42."

---

## Pre-Flight Contract

The builder exits non-zero before emitting a single cell if:
1. Any required CSV is missing.
2. Any expected column is absent from a CSV.
3. Any Sweep string literal used in code cells is not present in the real CSV data.
4. Any narrative `.md` file is missing.
5. Any phase aggregated CSV for §5 is missing.

---

## Notebook Cell Structure

| Section | Markdown cells | Code cells | Notes |
|---|---|---|---|
| Boilerplate | 0 | 3 | pip install, Drive mount, parser embed |
| §1: Feature Space | 2 | 2 | PCA scatter + homophily line plot |
| §2: τ=43 Anomaly | 2 | 2 | Drift dual-axis + prevalence bar |
| §3: Baselines | 1 | 1 | Efficiency scatter (log time × PRAUC) |
| §4: Grid/PCA | 1 | 1 | Grouped bar by K × Variation |
| §5: Deep Res MLP | 2 | **0** | Narrative + static phase table |
| §6: WF Trap | 1 | 1 | 3-model temporal PRAUC, regime shading |
| §7: Decay | 1 | 2 | SGC temporal + XGBoost λ-curve |
| **Total** | **10** | **12** | ~22 cells |

---

## Invariants the Implementation Must Not Violate

1. No Sweep string literal typed in any code cell (except those in `SWEEP_LITERALS`
   dict validated by pre-flight before emission).
2. PRAUC on y-axis of every plot; F1 may be mentioned in annotations but never as the
   main axis.
3. `Low_Confidence == True` → grey `axvspan` bands, always.
4. §6/§7 titles contain "n=1 seed, Seed=42".
5. §3/§4 include `prauc_std` error bars (3 seeds → real uncertainty).
6. §4 code cell ends with a `print()` disambiguating PCA-as-regularizer from
   PCA-as-drift-diagnostic (which appeared in §2).
7. Builder exits 1 on any missing file or column — never silently falls back.
