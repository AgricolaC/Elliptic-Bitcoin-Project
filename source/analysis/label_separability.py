"""
Propagated label-conditional separability — the broadcast-bias smoking gun.

The MMD canary measured ACROSS-timestep, WITHIN-label geometry on RAW features.
Broadcast-bias is a claim about WITHIN-timestep, ACROSS-label separability in
PROPAGATED space (S-hat^k X). This module measures that directly.

Decisive 2×2 readout at tau=43 (and tau=42 as pre-shock reference):

    raw separable? ×  prop separable?     interpretation
    ----------------------------------    ----------------------------------------
    raw YES / prop NO                     BROADCAST-BIAS CONFIRMED: propagation
                                          washes illicit nodes into the licit
                                          manifold; labels separable raw, not
                                          after S-hat^k aggregation.
    raw YES / prop YES                    NOT broadcast-bias. Representation is
                                          intact; failure is class imbalance at
                                          the classifier head. Soften the claim.
    raw NO  / prop NO                     Survivors intrinsically licit-like even
                                          before propagation. Different story.
    raw NO  / prop YES                    Propagation HELPS. Surprising; investigate.

"separable?" = size-matched MMD2 permutation test, p <= SEP_P_MAX.

CRITICAL: S-hat must match YOUR model. Set --k and --directed to your winning grid
config (K and Dir columns). S-hat^k X is parameter-free (only the final W is trained,
and W is NOT used here), so this is not circular. But a DIFFERENT operator measures a
different representation than the one whose collapse you are claiming — verify against
build_graph.py before trusting the verdict.

Normalization: D^{-1/2} (max(A,A^T) + I) D^{-1/2}
  This matches layers.py gcn_norm exactly: edge union (not sum), then self-loops,
  then symmetric normalisation. Using A + A^T here would double-count bidirectional
  edges and produce a different operator.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

HERE   = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(HERE)
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from analysis.tda_diagnostic import (
    _load_df, mmd2_unbiased,
    append_rows, RESULTS_DIR, FALSIFY_CSV, FALSIFY_COLS, TAU_RANGE,
)

# ── pre-registered constants (freeze before run) ───────────────────────────
SEP_P_MAX    = 0.05
N_PERM       = 1000
K_SEEDS      = 10
SHOCK_TAU    = 43
PRESHOCK_TAU = 42
LOW_CONF_N   = 30       # flag taus with fewer than this many illicit nodes

SEP_SWEEP = "BB: Label-Cond Separability Raw-vs-Prop"
SEP_CSV   = os.path.join(RESULTS_DIR, "label_separability.csv")
SEP_COLS  = [
    "Sweep", "Representation", "tau", "n_illicit", "n_matched",
    "seed", "mmd2", "perm_p", "separable", "Low_Confidence",
]


# ============================================================
# PROPAGATION OPERATOR  (must match the model's S-hat^k)
# ============================================================
def build_propagation_operator(directed: bool) -> tuple[sp.csr_matrix, dict]:
    """Return (S_hat, txid_to_row) over ALL nodes (labeled + unknown).

    directed=False: D^{-1/2} (max(A, A^T) + I) D^{-1/2}  — matches gcn_norm in layers.py.
    directed=True:  D^{-1} (A + I)  row-normalised (random-walk on out-edges).

    max(A, A^T) is the binary edge-union (not A + A^T which double-counts
    bidirectional edges). Since the Elliptic graph has no cross-temporal edges
    (enforced by _validate_temporal_edges), the global matrix is block-diagonal
    and gives identical per-timestep normalisation to per-timestep application.
    """
    df, _ = _load_df()
    from data.load_dataset import download_and_load_data
    _, df_edge, _, _ = download_and_load_data()
    src_col, dst_col = df_edge.columns[0], df_edge.columns[1]

    nodes       = df["txId"].values
    txid_to_row = {int(tx): i for i, tx in enumerate(nodes)}
    N           = len(nodes)

    e    = df_edge[[src_col, dst_col]].values
    keep = np.array([(int(s) in txid_to_row and int(d) in txid_to_row) for s, d in e])
    e    = e[keep]
    if len(e):
        rows_idx = np.fromiter((txid_to_row[int(s)] for s, _ in e), int, len(e))
        cols_idx = np.fromiter((txid_to_row[int(d)] for _, d in e), int, len(e))
        A = sp.csr_matrix((np.ones(len(e)), (rows_idx, cols_idx)), shape=(N, N))
    else:
        A = sp.csr_matrix((N, N))

    I = sp.identity(N, format="csr")
    if directed:
        Ahat = A + I
        deg  = np.asarray(Ahat.sum(axis=1)).ravel()
        dinv = sp.diags(1.0 / np.maximum(deg, 1e-12))
        S    = (dinv @ Ahat).tocsr()
    else:
        # binary union max(A, A^T): non-zero wherever either direction has an edge
        Asym     = A.maximum(A.T) + I          # ← edge union, NOT A + A^T
        deg      = np.asarray(Asym.sum(axis=1)).ravel()
        dinv_sq  = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        S        = (dinv_sq @ Asym @ dinv_sq).tocsr()
    return S, txid_to_row


def propagate(S: sp.csr_matrix, X: np.ndarray, k: int,
              multiscale: bool = True) -> np.ndarray:
    """Apply S-hat^k X.
    multiscale=True  → [X | S^1 X | ... | S^k X]  (matches use_multiscale_prop=True)
    multiscale=False → S^k X only
    k=0 returns X unchanged in both modes.
    """
    out = X.astype(np.float64)
    if multiscale:
        hops = [out]
        for _ in range(k):
            out = S @ out
            hops.append(out)
        return np.hstack(hops)
    else:
        for _ in range(k):
            out = S @ out
        return out


# ============================================================
# SIZE-MATCHED MMD2 PERMUTATION TEST
# ============================================================
def _mmd2_permutation_test(X: np.ndarray, Y: np.ndarray, sigma: float,
                            n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Observed MMD²(X,Y) and size-matched permutation p-value.
    Pool → repartition → fraction ≥ observed (continuity-corrected: never p=0).
    """
    m, n = len(X), len(Y)
    obs  = mmd2_unbiased(X, Y, sigma)
    Z    = np.vstack([X, Y])
    ge   = 0
    for _ in range(n_perm):
        idx = rng.permutation(m + n)
        if mmd2_unbiased(Z[idx[:m]], Z[idx[m:]], sigma) >= obs:
            ge += 1
    return float(obs), (ge + 1) / (n_perm + 1)


def _sigma_from_pool(Xs: np.ndarray, max_pts: int = 2000,
                     rng: np.random.Generator | None = None) -> float:
    """Median pairwise distance on a subsample of Xs (median heuristic)."""
    if rng is None:
        rng = np.random.default_rng(0)
    pool = Xs if len(Xs) <= max_pts else Xs[rng.choice(len(Xs), max_pts, replace=False)]
    d    = np.sqrt(np.maximum(
        ((pool[:, None, :] - pool[None, :, :]) ** 2).sum(axis=2), 0.0
    ))
    idx  = np.triu_indices(len(pool), k=1)
    med  = float(np.median(d[idx]))
    return med if med > 0 else 1.0


# ============================================================
# SEPARABILITY SERIES  (raw and propagated, all taus)
# ============================================================
def label_separability_series(k: int, directed: bool,
                               taus: list[int] = TAU_RANGE) -> None:
    """Per tau: size-matched illicit-vs-licit MMD2 + permutation p, for RAW and
    PROPAGATED representations.

    Each representation uses its OWN fixed sigma (median-heuristic on pooled
    labeled nodes at that scale). Read the TEMPORAL pattern WITHIN a representation
    (does prop-separability crater at tau=43 vs tau=42?); do NOT compare raw-MMD
    vs prop-MMD magnitudes directly — different sigma, different scales.
    """
    df, feature_cols = _load_df()
    # Pre-scaler: StandardScaler on labeled nodes only (leakage-free: fits on train
    # steps in the actual model, but for the diagnostic across all 49 taus we fit on
    # ALL labeled nodes to avoid artificially privileging any split).
    label = df["label"].values
    ts    = df["ts"].values
    lab_mask = (label == 1) | (label == 0)

    X_raw = df[feature_cols].values.astype(np.float64)
    scaler_raw = StandardScaler().fit(X_raw[lab_mask])
    X_scaled   = scaler_raw.transform(X_raw)

    print(f"Building propagation operator (directed={directed})...")
    S, txid_to_row = build_propagation_operator(directed)
    # align df rows → global adjacency rows
    row_of = df["txId"].map(txid_to_row)
    assert not row_of.isna().any(), "txId alignment failed: some nodes missing from adjacency"
    row_of = row_of.values.astype(int)

    print(f"Propagating S_hat^{k} X (multiscale={True}) over full graph...")
    X_prop_full = propagate(S, X_scaled, k, multiscale=True)
    # X_prop_full is indexed by global adjacency row; reorder to match df ordering
    X_prop = X_prop_full[row_of]   # shape (N_df, d*(k+1))

    prop_rep_name = f"Prop_k{k}_Dir{int(directed)}"
    reps = {
        "Raw":         X_scaled,
        prop_rep_name: X_prop,
    }

    rows_out: list[dict] = []
    for rep_name, X_all in reps.items():
        # fixed sigma for this representation, computed once on pooled labeled nodes
        sigma = _sigma_from_pool(X_all[lab_mask], max_pts=2000,
                                 rng=np.random.default_rng(0))
        print(f"\n{rep_name}: d={X_all.shape[1]}, sigma={sigma:.4f}")

        for tau in taus:
            ill_mask = (ts == tau) & (label == 1)
            lic_mask = (ts == tau) & (label == 0)
            ill = X_all[ill_mask]
            lic = X_all[lic_mask]
            n_ill = len(ill)
            if n_ill < 3 or len(lic) < 3:
                continue
            low_conf = int(n_ill < LOW_CONF_N)
            n_match  = min(n_ill, len(lic))

            for seed in range(K_SEEDS):
                rng = np.random.default_rng(seed)
                a = ill if n_ill  <= n_match else ill[rng.choice(n_ill, n_match, replace=False)]
                b = lic if len(lic) <= n_match else lic[rng.choice(len(lic), n_match, replace=False)]
                mmd2, p = _mmd2_permutation_test(a, b, sigma, N_PERM, rng)
                rows_out.append({
                    "Sweep": SEP_SWEEP, "Representation": rep_name,
                    "tau": tau, "n_illicit": n_ill, "n_matched": n_match,
                    "seed": seed, "mmd2": mmd2, "perm_p": p,
                    "separable": int(p <= SEP_P_MAX),
                    "Low_Confidence": low_conf,
                })
            print(f"  tau={tau:>2d}: n_ill={n_ill:>4d}  done", end="\r", flush=True)
        print(f"\n  {rep_name}: complete")

    append_rows(SEP_CSV, rows_out, SEP_COLS)
    print(f"\nWritten {len(rows_out)} rows to {SEP_CSV}")


# ============================================================
# BROADCAST-BIAS VERDICT  (the 2×2 at tau=43 vs tau=42)
# ============================================================
def evaluate_broadcast_bias(k: int, directed: bool) -> None:
    if not os.path.exists(SEP_CSV):
        print(f"No data at {SEP_CSV}. Run --action run first.")
        return
    df  = pd.read_csv(SEP_CSV)
    df  = df[df.Sweep == SEP_SWEEP]
    prop_rep = f"Prop_k{k}_Dir{int(directed)}"

    def sep_stats(rep: str, tau: int) -> tuple[float, float, float, int]:
        s = df[(df.Representation == rep) & (df.tau == tau)]
        if len(s) == 0:
            return (np.nan, np.nan, np.nan, 0)
        return (float(s.separable.mean()), float(s.perm_p.median()),
                float(s.mmd2.median()), int(s.n_illicit.iloc[0]))

    print(f"\n{'='*68}")
    print(f"BROADCAST-BIAS 2×2  (config K={k}, directed={directed})")
    print(f"  'separable' = perm p <= {SEP_P_MAX} in majority of seeds (>50%)")
    print(f"  Read WITHIN-rep temporal change (tau=42→43), not across-rep magnitudes")
    print(f"{'='*68}")

    summary = {}
    for tau in (PRESHOCK_TAU, SHOCK_TAU):
        rf, rp, rm, n = sep_stats("Raw", tau)
        pf, pp, pm, _ = sep_stats(prop_rep, tau)
        raw_sep  = rf >= 0.5 if not np.isnan(rf) else None
        prop_sep = pf >= 0.5 if not np.isnan(pf) else None
        lc_note  = "  [Low-confidence: n_ill<30]" if n < LOW_CONF_N else ""
        print(f"\n  tau={tau} (n_illicit={n}){lc_note}")
        print(f"    Raw:  sep={raw_sep}  frac={rf:.0%}  median_p={rp:.4f}  median_mmd2={rm:.4f}")
        print(f"    Prop: sep={prop_sep} frac={pf:.0%}  median_p={pp:.4f}  median_mmd2={pm:.4f}")
        summary[tau] = (raw_sep, prop_sep, rf, pf, n)

    raw43, prop43, rf43, pf43, n43 = summary[SHOCK_TAU]
    raw42, prop42, rf42, pf42, n42 = summary[PRESHOCK_TAU]

    print(f"\n  Pre-shock reference (tau={PRESHOCK_TAU}): Raw={raw42}, Prop={prop42}")
    print(f"  Shock (tau={SHOCK_TAU}, n={n43}):        Raw={raw43}, Prop={prop43}")

    if raw43 is True and prop43 is False:
        verdict = (
            "BROADCAST_BIAS_CONFIRMED: raw-separable at tau=43 but prop-INSEPARABLE. "
            "SGC neighbourhood aggregation washes the 24 illicit nodes into their "
            "licit-dominated neighbours; the head receives a collapsed representation."
        )
    elif raw43 is True and prop43 is True:
        verdict = (
            "NOT_BROADCAST_BIAS: both raw and prop separable at tau=43. "
            "The representation survives propagation; the failure is class imbalance "
            "at the classifier head. Soften the broadcast-bias framing to 'head-level "
            "imbalance' rather than 'representational collapse'."
        )
    elif raw43 is False and prop43 is False:
        verdict = (
            "INTRINSIC_LICIT_LIKE: tau=43 illicit survivors are raw-inseparable from "
            "licit nodes even before propagation. Not a propagation effect."
        )
    elif raw43 is False and prop43 is True:
        verdict = (
            "PROP_HELPS: propagation increases separability at tau=43. Unexpected; investigate."
        )
    else:
        verdict = f"INCONCLUSIVE: raw={raw43}, prop={prop43} — insufficient data or power."

    note = (
        f"\n  CAUTION: n_illicit={n43} at tau=43 is low-power. "
        "A non-significant prop cell is 'underpowered/inconclusive', NOT 'proven inseparable'. "
        f"Use the tau={PRESHOCK_TAU} reference (n={n42}) as the within-series power anchor."
    )
    print(f"\n  VERDICT: {verdict}")
    print(note)
    print("=" * 68)

    append_rows(FALSIFY_CSV, [{
        "Sweep":           SEP_SWEEP,
        "hypothesis":      "broadcast_bias_2x2",
        "statistic_name":  f"raw_sep{int(raw43 or False)}_prop_sep{int(prop43 or False)}",
        "statistic_value": pf43,
        "threshold":       SEP_P_MAX,
        "seed_fraction":   rf43,
        "verdict":         verdict,
    }], FALSIFY_COLS)
    print(f"Falsification log updated → {FALSIFY_CSV}")


# ============================================================
# CORRECTED CANARY VERDICT HELPER
# ============================================================
def corrected_canary_verdict(b_perm_p: float, b_median_z: float,
                              b_z_seed_frac: float, b_rank: int,
                              a_fires: bool) -> str:
    """Significance-based B-fires gate. Rank is a DESCRIPTOR, not the fire/silent switch.
    The rank-1 gate belongs in the PH arm's topological-exceptionality test, not here.
    """
    b_fires = (b_perm_p <= SEP_P_MAX) and (b_median_z >= 3.0) and (b_z_seed_frac >= 0.8)
    exceptional = (b_rank == 1)
    if a_fires and b_fires:
        world = "A_fires / B_fires"
    elif (not a_fires) and b_fires:
        world = (
            "A-silent / B-FIRES (significance-based). "
            f"tau=43 shows a real geometric shift in the illicit subpopulation "
            f"(perm p={b_perm_p:.4f}, median Z={b_median_z:.2f}, "
            f"{b_z_seed_frac:.0%} seeds pass) but is not the single largest "
            f"illicit-subpop transition (rank {b_rank}, exceptional={exceptional}). "
            "Skip PH not because B is silent, but because N=24 is underpowered for "
            "clean H1 topology signatures (Vaccarino-style requires >~300 sample points)."
        )
    elif (not a_fires) and (not b_fires):
        world = "A-silent / B-silent: clean World gamma (label-prevalence event only)."
    else:
        world = "A-fires / B-silent: all-node manifold shifts without illicit subpop signal."
    return world


# ============================================================
# SMOKE TEST
# ============================================================
def smoke() -> None:
    """No real data. Validates propagation operator and permutation plumbing."""
    rng = np.random.default_rng(0)

    # 6-node path graph
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5)]
    rows_e = [s for s,d in edges]; cols_e = [d for s,d in edges]
    A  = sp.csr_matrix((np.ones(len(edges)), (rows_e, cols_e)), shape=(6,6))
    I  = sp.identity(6, format="csr")
    Asym    = A.maximum(A.T) + I
    deg     = np.asarray(Asym.sum(axis=1)).ravel()
    dinv_sq = sp.diags(1.0 / np.sqrt(deg))
    S  = (dinv_sq @ Asym @ dinv_sq).tocsr()

    X   = np.eye(6)
    X1  = propagate(S, X, 1, multiscale=False)
    Xms = propagate(S, X, 1, multiscale=True)
    assert not np.allclose(X1, X),              "propagation must mix neighbours"
    assert Xms.shape == (6, 12),                f"multiscale shape wrong: {Xms.shape}"
    assert np.allclose(Xms[:, :6], X),          "multiscale must prefix with raw X"
    assert np.allclose(Xms[:, 6:], X1),         "multiscale last block must be S^1 X"
    assert np.isfinite(S.data).all(),            "operator has non-finite entries"

    # permutation test: identical distributions → p high; separated → p low
    Xc = rng.normal(0, 1, (40, 5))
    Yc = rng.normal(0, 1, (40, 5))
    Yf = rng.normal(6, 1, (40, 5))
    _, p_same = _mmd2_permutation_test(Xc, Yc, 1.0, 200, rng)
    _, p_diff = _mmd2_permutation_test(Xc, Yf, 1.0, 200, rng)
    assert p_same > 0.2,   f"p_same too low ({p_same:.3f}) — perm test may be broken"
    assert p_diff < 0.05,  f"p_diff too high ({p_diff:.3f}) — perm test may be broken"

    print("smoke PASS (operator shape, multiscale concat, permutation plumbing)")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Broadcast-bias label-separability measurement")
    ap.add_argument("--action",   choices=["smoke", "run", "evaluate"], required=True)
    ap.add_argument("--k",        type=int, default=1,
                    help="Propagation hops — must match your grid's K (default: 1)")
    ap.add_argument("--directed", action="store_true",
                    help="Set iff your grid's Dir=T (default: symmetric)")
    a = ap.parse_args()

    if a.action == "smoke":
        smoke()
    elif a.action == "run":
        label_separability_series(k=a.k, directed=a.directed)
    elif a.action == "evaluate":
        evaluate_broadcast_bias(k=a.k, directed=a.directed)
