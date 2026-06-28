"""
sweep_parser.py — Parse Elliptic-project Sweep identifier strings.

Real string grammar (verified against actual CSV data 2026-06-27):
  Ablation: Decay λ=X on XGBoost
  Ablation: Decay λ=X on N Dir Topo Var      (N=1/2/3, Dir=T/F, Var=Base/PCA)
  Ablation: IPCA N Dir Topo
  Baseline: Name (args)
  Sweep N: SGC (baseline) [(Seed N, Var X)]
  Sweep N: + MLP Head [(Seed N, Var X)]
  Grid: K=N, Dir=X, Topo=Y [(Seed N, Var X)] [WF]
  Best WF: Grid: K=N, Dir=X, Topo=Y [(Seed N, Var X)]
  Best WF: Sweep N: ... [(Seed N, Var X)]
  Ensemble: ...
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass(frozen=True)
class SweepInfo:
    raw: str
    family: Optional[str]       # SGC | SGC+MLP | IPCA | XGBoost | RandomForest |
                                 # IsolationForest | LogisticRegression | GCN | Ensemble
    family_tag: Optional[str]   # Ablation | Baseline | Grid | Ensemble
    lam: Optional[float]        # decay λ; None = no decay
    K: Optional[int]
    Dir: Optional[bool]
    Topo: Optional[str]         # 'None' | 'early' | 'late'  (string, not Python None)
    seed: Optional[int]
    variation: Optional[str]    # 'Base' | 'PCA'

    @property
    def is_decay(self) -> bool:
        return self.lam is not None

    def as_dict(self) -> dict:
        return asdict(self)

# Regexes for explicit "K=N, Dir=X, Topo=Y" encoding (Grid strings)
_K_EXPLICIT    = re.compile(r"\bK\s*=\s*([0-9]+)")
_DIR_EXPLICIT  = re.compile(r"\bDir\s*=\s*([TF])", re.IGNORECASE)
_TOPO_EXPLICIT = re.compile(r"\bTopo\s*=\s*(None|early|late)", re.IGNORECASE)
# Seed/Variation embedded in parens
_SEED_RE = re.compile(r"Seed\s+([0-9]+)")
_VAR_RE  = re.compile(r"\bVar\s+([A-Za-z0-9]+)")
# λ value (unicode λ only — project convention)
_LAM_RE  = re.compile(r"λ\s*=\s*([0-9]*\.?[0-9]+)")
# "on N Dir Topo Var" in Ablation: Decay strings
_DECAY_SGC_RE = re.compile(
    r"\bon\s+([123])\s+([TF])\s+(early|late|None)\s+(Base|PCA)", re.IGNORECASE)
# "IPCA N Dir Topo" positional encoding
_IPCA_BODY_RE = re.compile(
    r"IPCA\s+([123])\s+([TF])\s+(early|late|None)", re.IGNORECASE)


def _dir(s: str) -> bool:
    return s.upper() == "T"


def parse_sweep(s: str) -> SweepInfo:
    """Parse a single Sweep identifier string into a SweepInfo."""
    if not isinstance(s, str):
        raise TypeError(f"expected str, got {type(s).__name__}: {s!r}")
    s = s.strip()

    lam_m = _LAM_RE.search(s)
    lam = float(lam_m.group(1)) if lam_m else None
    seed_m = _SEED_RE.search(s)
    seed = int(seed_m.group(1)) if seed_m else None
    var_m = _VAR_RE.search(s)
    variation = var_m.group(1) if var_m else None

    def _make(**kw):
        base = dict(raw=s, lam=lam, seed=seed, variation=variation,
                    family=None, family_tag=None, K=None, Dir=None, Topo=None)
        base.update(kw)
        return SweepInfo(**base)

    # Ablation: Decay λ=X on BODY
    if s.startswith("Ablation: Decay"):
        if "on XGBoost" in s:
            return _make(family="XGBoost", family_tag="Ablation")
        m = _DECAY_SGC_RE.search(s)
        if m:
            return _make(family="SGC", family_tag="Ablation",
                         K=int(m.group(1)), Dir=_dir(m.group(2)),
                         Topo=m.group(3), variation=m.group(4))
        # Fallback to K=N, Dir=X, Topo=Y explicit encoding
        k_m = _K_EXPLICIT.search(s)
        d_m = _DIR_EXPLICIT.search(s)
        t_m = _TOPO_EXPLICIT.search(s)
        return _make(family="SGC", family_tag="Ablation",
                     K=int(k_m.group(1)) if k_m else None,
                     Dir=_dir(d_m.group(1)) if d_m else None,
                     Topo=t_m.group(1) if t_m else None)

    # Ablation: IPCA N Dir Topo
    if s.startswith("Ablation: IPCA"):
        m = _IPCA_BODY_RE.search(s)
        if m:
            return _make(family="IPCA", family_tag="Ablation",
                         K=int(m.group(1)), Dir=_dir(m.group(2)), Topo=m.group(3))
        # Fallback to K=N, Dir=X, Topo=Y explicit encoding
        k_m = _K_EXPLICIT.search(s)
        d_m = _DIR_EXPLICIT.search(s)
        t_m = _TOPO_EXPLICIT.search(s)
        return _make(family="IPCA", family_tag="Ablation",
                     K=int(k_m.group(1)) if k_m else None,
                     Dir=_dir(d_m.group(1)) if d_m else None,
                     Topo=t_m.group(1) if t_m else None)

    # Baseline: Name
    if s.startswith("Baseline:"):
        fam = None
        if "XGBoost" in s or "XGB" in s:      fam = "XGBoost"
        elif "RandomForest" in s:              fam = "RandomForest"
        elif "IsolationForest" in s:           fam = "IsolationForest"
        elif "Logistic Regression" in s:       fam = "LogisticRegression"
        elif "GCN" in s:                       fam = "GCN"
        return _make(family=fam, family_tag="Baseline")

    # Sweep N: ... or Best WF: Sweep N: ...
    if re.search(r"(?:^|Best WF:\s*)Sweep\s+\d+:", s):
        fam = "SGC+MLP" if ("+ MLP Head" in s or "MLP Head" in s) else "SGC"
        return _make(family=fam)

    # Grid: K=N, Dir=X, Topo=Y ... or Best WF: Grid: K=N ...
    if "Grid:" in s:
        k_m = _K_EXPLICIT.search(s)
        d_m = _DIR_EXPLICIT.search(s)
        t_m = _TOPO_EXPLICIT.search(s)
        return _make(
            family_tag="Grid",
            K=int(k_m.group(1)) if k_m else None,
            Dir=_dir(d_m.group(1)) if d_m else None,
            Topo=t_m.group(1) if t_m else None,
        )

    # Ensemble
    if s.startswith("Ensemble:"):
        return _make(family="Ensemble", family_tag="Ensemble")

    return _make()


def family_config_id(s: str) -> str:
    """Strip embedded '(Seed N, Var X)' / '(Var X)' so configs group across seeds."""
    return re.sub(r"\s*\((?:Seed\s+\d+,\s*)?Var\s+[^)]+\)", "", s).strip()


# ── pandas helpers ────────────────────────────────────────────────────────────
_IGNORE = object()


EXPLICIT_COL_MAP = {
    "lam":  "Decay_Lambda",
    "K":    "SGC_K",
    "Dir":  "Directionality",    # bool in CSV; coerce to bool
    "Topo": "Topological_Injection",
}

def add_parsed_columns(df, sweep_col: str = "Sweep", prefix: str = "_"):
    """Return copy of df with parsed SweepInfo fields as prefixed columns."""
    import pandas as pd
    info = df[sweep_col].map(parse_sweep)
    out = df.copy()
    for field in ("family", "family_tag", "lam", "K", "Dir", "Topo", "seed", "variation"):
        out[f"{prefix}{field}"] = info.map(lambda i, f=field: getattr(i, f))
        
    # Column-first override
    for field, col in EXPLICIT_COL_MAP.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce") if field in ("lam", "K") else df[col]
            mask = vals.notna() & (vals != "") & (vals != "N/A")
            out.loc[mask, f"{prefix}{field}"] = vals[mask]

    return out


def select(df, *, family=_IGNORE, family_tag=_IGNORE, K=_IGNORE, Dir=_IGNORE,
           Topo=_IGNORE, lam=_IGNORE, decay=_IGNORE, sweep_col: str = "Sweep"):
    """Return boolean Series matching the given constraints (omitted fields = ignored).

    Examples:
        df[select(df, family="XGBoost", decay=False)]   # baseline XGBoost WF
        df[select(df, family="SGC", K=2, Dir=True, lam=0.25)]
        df[select(df, family_tag="Grid", K=3)]
    """
    import pandas as pd
    info = df[sweep_col].map(parse_sweep)
    mask = pd.Series(True, index=df.index)

    def _eq(attr, val):
        return info.map(lambda i, a=attr, v=val: getattr(i, a) == v)

    if family is not _IGNORE:     mask &= _eq("family", family)
    if family_tag is not _IGNORE: mask &= _eq("family_tag", family_tag)
    if K is not _IGNORE:          mask &= _eq("K", K)
    if Dir is not _IGNORE:        mask &= _eq("Dir", Dir)
    if Topo is not _IGNORE:       mask &= _eq("Topo", Topo)
    if lam is not _IGNORE:        mask &= _eq("lam", lam)
    if decay is not _IGNORE:      mask &= info.map(lambda i: i.is_decay == decay)
    return mask
