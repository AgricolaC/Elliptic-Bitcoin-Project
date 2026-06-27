"""Tests sweep_parser against REAL Sweep strings from the project CSVs.
All test cases were verified to exist in walk_forward_timesteps.csv,
sweep_results.csv, or final_aggregated_results.csv.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add source/reporting to path so we can import sweep_parser
sys.path.insert(0, str(Path(__file__).parent.parent / "source" / "reporting"))
from sweep_parser import parse_sweep, select, add_parsed_columns, family_config_id

# ---------------------------------------------------------------------------
# Parse-field cases: (raw_string, field_name, expected_value)
# ---------------------------------------------------------------------------
PARSE_CASES = [
    # Ablation: Decay — SGC family (K/Dir/Topo encoded positionally)
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "family",     "SGC"),
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "family_tag", "Ablation"),
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "lam",        0.05),
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "K",          3),
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "Dir",        False),
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "Topo",       "late"),
    ("Ablation: Decay λ=0.05 on 3 F late Base",  "variation",  "Base"),
    ("Ablation: Decay λ=0.25 on 2 T early Base", "K",          2),
    ("Ablation: Decay λ=0.25 on 2 T early Base", "Dir",        True),
    ("Ablation: Decay λ=0.25 on 2 T early Base", "Topo",       "early"),
    ("Ablation: Decay λ=0.5 on 3 T None PCA",    "Topo",       "None"),
    ("Ablation: Decay λ=0.5 on 3 T None PCA",    "variation",  "PCA"),
    # Ablation: Decay — XGBoost
    ("Ablation: Decay λ=0.5 on XGBoost",         "family",     "XGBoost"),
    ("Ablation: Decay λ=0.5 on XGBoost",         "lam",        0.5),
    ("Ablation: Decay λ=0.05 on XGBoost",        "lam",        0.05),
    ("Ablation: Decay λ=0.25 on XGBoost",        "lam",        0.25),
    # Ablation: IPCA (positional encoding, no K=/Dir= tokens)
    ("Ablation: IPCA 3 F late",                  "family",     "IPCA"),
    ("Ablation: IPCA 3 F late",                  "family_tag", "Ablation"),
    ("Ablation: IPCA 3 F late",                  "K",          3),
    ("Ablation: IPCA 3 F late",                  "Dir",        False),
    ("Ablation: IPCA 3 F late",                  "Topo",       "late"),
    ("Ablation: IPCA 3 T None",                  "Dir",        True),
    ("Ablation: IPCA 3 T None",                  "Topo",       "None"),
    # Baseline: explicit family_tag + family detection
    ("Baseline: XGBoost WF (epsilon-fallback)",  "family",     "XGBoost"),
    ("Baseline: XGBoost WF (epsilon-fallback)",  "family_tag", "Baseline"),
    ("Baseline: XGBoost WF (epsilon-fallback)",  "lam",        None),
    ("Baseline: RandomForest (166)",             "family",     "RandomForest"),
    ("Baseline: IsolationForest (166)",          "family",     "IsolationForest"),
    ("Baseline: Logistic Regression (166)",      "family",     "LogisticRegression"),
    ("Baseline: PyG GCN (2-layer)",              "family",     "GCN"),
    # Sweep N: (SGC and SGC+MLP, with and without seed suffix)
    ("Sweep 1: SGC (baseline)",                  "family",     "SGC"),
    ("Sweep 1: SGC (baseline) (Seed 42, Var Base)", "family",  "SGC"),
    ("Sweep 1: SGC (baseline) (Seed 42, Var Base)", "seed",    42),
    ("Sweep 1: SGC (baseline) (Seed 42, Var Base)", "variation", "Base"),
    ("Sweep 2: + MLP Head",                      "family",     "SGC+MLP"),
    ("Sweep 2: + MLP Head (Seed 42, Var Base)",  "family",     "SGC+MLP"),
    ("Sweep 2: + MLP Head (Seed 42, Var PCA)",   "variation",  "PCA"),
    # Grid: explicit K=/Dir=/Topo= tokens
    ("Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "family_tag", "Grid"),
    ("Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "K",          2),
    ("Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "Dir",        False),
    ("Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "Topo",       "None"),
    ("Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "variation",  "Base"),
    ("Grid: K=3, Dir=T, Topo=early (Seed 42, Var PCA)", "Dir",        True),
    ("Grid: K=3, Dir=T, Topo=early (Seed 42, Var PCA)", "variation",  "PCA"),
    ("Grid: K=3, Dir=T, Topo=early (Var Base) WF",      "variation",  "Base"),
    # Best WF: Grid: — "Best WF:" prefix must NOT break Grid detection
    ("Best WF: Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "family_tag", "Grid"),
    ("Best WF: Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "K",          2),
    ("Best WF: Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)", "Dir",        False),
    # Best WF: Sweep N: — family detection through "Best WF:" prefix
    ("Best WF: Sweep 1: SGC (baseline) (Seed 42, Var Base)", "family", "SGC"),
    ("Best WF: Sweep 2: + MLP Head (Seed 42, Var Base)",     "family", "SGC+MLP"),
    # Ensemble
    ("Ensemble: XGB(0.7) + 3 T late PCA(0.3)",  "family",     "Ensemble"),
    ("Ensemble: XGB(0.7) + 3 T late PCA(0.3)",  "family_tag", "Ensemble"),
]

@pytest.mark.parametrize("raw,field,expected", PARSE_CASES)
def test_parse_field(raw, field, expected):
    result = getattr(parse_sweep(raw), field)
    assert result == expected, (
        f"parse_sweep({raw!r})[{field!r}]: expected {expected!r}, got {result!r}"
    )

def test_is_decay_true():
    assert parse_sweep("Ablation: Decay λ=0.25 on 2 T early Base").is_decay is True

def test_is_decay_false():
    assert parse_sweep("Baseline: XGBoost WF (epsilon-fallback)").is_decay is False

def test_family_config_id_strips_seed():
    assert family_config_id(
        "Grid: K=2, Dir=F, Topo=None (Seed 43, Var Base)"
    ) == "Grid: K=2, Dir=F, Topo=None"

def test_family_config_id_strips_var_only():
    assert family_config_id(
        "Grid: K=3, Dir=T, Topo=early (Var Base) WF"
    ) == "Grid: K=3, Dir=T, Topo=early WF"

# ---------------------------------------------------------------------------
# select() integration tests on a mini-DataFrame
# ---------------------------------------------------------------------------
MINI_DF = pd.DataFrame({"Sweep": [
    "Baseline: XGBoost WF (epsilon-fallback)",
    "Ablation: Decay λ=0.25 on XGBoost",
    "Ablation: Decay λ=0.5 on XGBoost",
    "Ablation: Decay λ=0.25 on 3 T late PCA",
    "Sweep 2: + MLP Head (Seed 42, Var Base)",
    "Grid: K=3, Dir=T, Topo=late (Seed 42, Var PCA)",
    "Best WF: Grid: K=2, Dir=F, Topo=None (Seed 42, Var Base)",
    "Best WF: Sweep 2: + MLP Head (Seed 42, Var Base)",
]})

@pytest.mark.parametrize("kwargs,expected_count", [
    ({"family": "XGBoost", "decay": False}, 1),
    ({"family": "XGBoost"},                 3),
    ({"family": "SGC+MLP"},                 2),
    ({"family": "SGC", "lam": 0.25},        1),
    ({"family_tag": "Grid", "K": 3},        1),
    ({"family_tag": "Grid", "K": 2},        1),
    ({"family_tag": "Grid"},                2),
    ({"decay": True},                       3),
    ({"lam": 0.25},                         2),
])
def test_select(kwargs, expected_count):
    mask = select(MINI_DF, **kwargs)
    assert mask.sum() == expected_count, (
        f"select({kwargs}) expected {expected_count} rows, got {mask.sum()}"
    )

def test_add_parsed_columns():
    df = add_parsed_columns(MINI_DF)
    assert "_family" in df.columns
    assert "_lam" in df.columns
    assert "_K" in df.columns
    # SGC+MLP rows should have _family == "SGC+MLP"
    mlp_rows = df[df["_family"] == "SGC+MLP"]
    assert len(mlp_rows) == 2
