"""
Topological diagnostic of the tau=43 regime shift on Elliptic.

Tests whether a persistent-homology summary of the per-snapshot node point cloud
localizes the tau=43 shift more sharply and robustly than MMD on the same clouds.
Design mirrors Vaccarino et al. (PRE 2016 — Donato, Gori, Pettini, Petri, De Nigris,
Franzosi, Vaccarino): MFXY model = positive control (topology fires where a transition
exists); phi^4 = negative control (no finite-N signature). Our phi^4 analog is the
temporal-shuffle null (H3). We do NOT claim the shift is a topological transition
(no Franzosi-Pettini theorem applies to Elliptic).

PIPELINE (action order)
-----------------------
1. --action smoke   : validate MMD plumbing; TDA plumbing if library is installed.
2. --action run     : MMD-only canary, dual-cloud (Cloud A = all nodes, Cloud B = illicit
                      only), tau=1..49. Runnable with only sklearn installed.
3. --action evaluate: read diagnostics CSV, compute robust Z, emit decision matrix and
                      falsification_log.csv entry.

Only after evaluate shows a geometric target (A or B fires) does the full PH arm
(--action run --ph) become worth the giotto-tda / ripser install.

PRE-REGISTERED CONSTANTS (spec sections 1-6)
Do NOT modify after the first run. Verdict-time assert checks they are unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# ── Path setup (script lives at source/analysis/tda_diagnostic.py) ──────────
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(HERE)          # source/
REPO_ROOT = os.path.dirname(SOURCE)
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

# ============================================================
# PRE-REGISTERED CONSTANTS (frozen before run)
# ============================================================
SHOCK_TAU = 43
TRANSITION_OF_INTEREST = (42, 43)          # T* = tau→tau+1 where tau=42
GUARD = {41, 42, 43, 44}                   # transitions incident to these are null-excluded
TAU_RANGE = list(range(1, 50))             # full 1..49 (spec §5 extension for robust MAD)

M_CAP = 800                                # Cloud A identical-M cap
K_SEEDS = 10
MAXDIM = 1                                 # H0 + H1 only
ME_PERCENTILE = 90
B_SHUFFLE = 200

# Decision thresholds — DO NOT tune post hoc
H1_Z_MIN = 3.0
H1_SEED_FRAC = 0.8          # >= 8/10 seeds must satisfy H1 Z gate
H2_MARGIN = 0.5
H2_SEED_FRAC = 0.7
H3_P_MAX = 0.01

# Cloud B (illicit-only) identical-M floor and significance parameters
M_B = 24            # fixed identical M; taus with N_illicit < M_B are excluded, not used at tiny M
N_PERM_B = 1000     # permutations for size-matched significance test at T*
B_PERM_P_MAX = 0.05 # pre-registered significance threshold for permutation p-value

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DIAG_CSV = os.path.join(RESULTS_DIR, "topological_diagnostics.csv")
FALSIFY_CSV = os.path.join(RESULTS_DIR, "falsification_log.csv")

DIAG_COLS = [
    "Sweep", "Variation", "tau", "seed", "shuffle_flag", "M",
    "beta0_total_pers", "beta1_total_pers", "beta1_max_life",
    "pers_entropy_h0", "pers_entropy_h1",
    "wasserstein_h0_to_next", "wasserstein_h1_to_next", "wasserstein_combined_to_next",
    "mmd2_to_next", "Low_Confidence",
]
FALSIFY_COLS = [
    "Sweep", "hypothesis", "statistic_name", "statistic_value",
    "threshold", "seed_fraction", "verdict",
]

MMD_SWEEP = "TDA: MMD Canary RawFeat Euclid tau=1-49"


# ============================================================
# DATA LAYER
# ============================================================
_cached_data: tuple | None = None


def _load_df() -> tuple[pd.DataFrame, list[str]]:
    """Load Elliptic data once and cache. Returns (df, feature_cols)."""
    global _cached_data
    if _cached_data is None:
        from data.load_dataset import download_and_load_data
        df, _, _, feature_cols = download_and_load_data()
        _cached_data = (df, feature_cols)
    return _cached_data


def load_point_clouds(taus: list[int]) -> dict[int, np.ndarray]:
    """Cloud A: all nodes (labeled + unknown) at each tau. Raw, unscaled."""
    df, feature_cols = _load_df()
    return {
        t: df[df.ts == t][feature_cols].values.astype(np.float32)
        for t in taus if (df.ts == t).any()
    }


def load_illicit_clouds(taus: list[int]) -> dict[int, np.ndarray]:
    """Cloud B: illicit-labeled nodes only (label == 1) at each tau."""
    df, feature_cols = _load_df()
    return {
        t: df[(df.ts == t) & (df.label == 1)][feature_cols].values.astype(np.float32)
        for t in taus if ((df.ts == t) & (df.label == 1)).any()
    }


# ============================================================
# CORE: standardizer, subsampling, sigma, MMD
# ============================================================

def fit_global_standardizer(clouds: dict[int, np.ndarray]) -> StandardScaler:
    """Global z-score fitted ONCE on pooled features across all tau. Per-snapshot fit is forbidden."""
    pooled = np.vstack([clouds[t] for t in sorted(clouds)])
    return StandardScaler().fit(pooled)


def effective_M(clouds: dict[int, np.ndarray]) -> int:
    return int(min(M_CAP, min(c.shape[0] for c in clouds.values())))


def subsample(X: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= M:
        return X
    return X[rng.choice(X.shape[0], size=M, replace=False)]


def median_heuristic_sigma(std_clouds_M: dict[int, np.ndarray]) -> float:
    """ONE fixed sigma on pooled standardized features. Never recomputed per pair."""
    pooled = np.vstack([std_clouds_M[t] for t in sorted(std_clouds_M)])
    rng = np.random.default_rng(0)
    if pooled.shape[0] > 2000:
        pooled = pooled[rng.choice(pooled.shape[0], 2000, replace=False)]
    d = pairwise_distances(pooled)
    med = float(np.median(d[np.triu_indices_from(d, k=1)]))
    return med if med > 0 else 1.0


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    g = 1.0 / (2.0 * sigma * sigma)
    Kxx = np.exp(-g * pairwise_distances(X, X, squared=True))
    Kyy = np.exp(-g * pairwise_distances(Y, Y, squared=True))
    Kxy = np.exp(-g * pairwise_distances(X, Y, squared=True))
    m, n = X.shape[0], Y.shape[0]
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(
        Kxx.sum() / (m * (m - 1))
        + Kyy.sum() / (n * (n - 1))
        - 2.0 * Kxy.mean()
    )


# ============================================================
# STATISTICS / HYPOTHESES
# ============================================================

def permutation_test_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    n_perm: int = N_PERM_B,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """
    Size-matched permutation significance test for MMD2(X, Y).

    Pools X and Y, repeatedly repartitions into subsets of sizes (|X|, |Y|),
    recomputes unbiased MMD2 each time.  Returns (observed_mmd2, p_value).

    This is the standard MMD significance test (Gretton et al. 2012).
    Unlike the cross-transition robust-Z, it is immune to M-vs-variance confound:
    every permuted resample has the same (|X|, |Y|) split.  B "fires" only if
    this p-value is significant AND the fixed-M_B Z holds.
    """
    obs = mmd2_unbiased(X, Y, sigma)
    rng = np.random.default_rng(rng_seed)
    pooled = np.vstack([X, Y])
    m = len(X)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        null[i] = mmd2_unbiased(pooled[idx[:m]], pooled[idx[m:]], sigma)
    p = float(np.mean(null >= obs))
    return obs, p


def robust_z(value: float, null_vals: np.ndarray) -> float:
    m = np.median(null_vals)
    mad = 1.4826 * np.median(np.abs(null_vals - m))
    return float((value - m) / mad) if mad > 0 else np.inf


def null_transitions(taus: list[int]) -> list[int]:
    """Consecutive transitions whose endpoints are both outside GUARD."""
    return [t for t in taus[:-1] if t not in GUARD and (t + 1) not in GUARD]


# ============================================================
# CSV I/O (append-only, schema-checked)
# ============================================================

def append_rows(path: str, rows: list[dict], cols: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# ============================================================
# SMOKE TEST (write-failing-first gate)
# ============================================================

def smoke_test() -> None:
    """
    Validate plumbing before any result is trusted.
    MMD section runs with only sklearn. TDA section runs only if a library is present.
    """
    rng = np.random.default_rng(0)
    A  = rng.normal(0, 1, size=(200, 8))
    A2 = A.copy()
    B  = rng.normal(8, 1, size=(200, 8))
    sigma = 1.0

    # MMD: identical -> ~0; separated -> large
    assert mmd2_unbiased(A, A2, sigma) < 1e-6, "MMD of identical clouds must be ~0"
    assert mmd2_unbiased(A, B, sigma) > mmd2_unbiased(A, A2, sigma), "MMD must separate"
    print("smoke_test MMD PASS")

    # Global standardizer: fitted once on pooled
    clouds = {0: A, 1: B}
    sc = fit_global_standardizer(clouds)
    pooled = np.vstack([A, B])
    np.testing.assert_allclose(sc.mean_, pooled.mean(axis=0), rtol=1e-5)
    print("smoke_test standardizer PASS")

    # Robust Z: signal >> null
    null = rng.normal(0, 1, size=30)
    z = robust_z(10.0, null)
    assert z > 5.0, f"robust_z should be large for clear outlier, got {z}"
    print("smoke_test robust_z PASS")

    # TDA: try giotto first, then ripser
    _tda_smoke(A, A2, B)


def _tda_smoke(A: np.ndarray, A2: np.ndarray, B: np.ndarray) -> None:
    try:
        from gtda.homology import VietorisRipsPersistence
        from gtda.diagrams import PairwiseDistance
        vr = VietorisRipsPersistence(
            homology_dimensions=(0, 1), max_edge_length=20.0, infinity_values=20.0
        )
        dg = vr.fit_transform(np.stack([A, A2, B]))
        # PairwiseDistance API: verify on your installed gtda version
        pd_ = PairwiseDistance(metric="wasserstein", order=np.inf).fit_transform(dg)
        assert pd_[0, 1] < pd_[0, 2], "PH Wasserstein must separate identical vs distant"
        print("smoke_test TDA (giotto-tda) PASS")
        return
    except ImportError:
        pass

    try:
        import ripser
        import persim
        dg_A  = ripser.ripser(A,  maxdim=1)["dgms"]
        dg_A2 = ripser.ripser(A2, maxdim=1)["dgms"]
        dg_B  = ripser.ripser(B,  maxdim=1)["dgms"]
        w_same = sum(persim.wasserstein(dg_A[d], dg_A2[d]) for d in range(2))
        w_diff = sum(persim.wasserstein(dg_A[d], dg_B[d])  for d in range(2))
        assert w_same < w_diff, "PH Wasserstein must separate identical vs distant"
        print("smoke_test TDA (ripser + persim) PASS")
        return
    except ImportError:
        pass

    print("smoke_test TDA SKIP — no giotto-tda or ripser installed (MMD arm still valid)")


# ============================================================
# MMD CANARY ORCHESTRATION  (steps 1-2; no TDA library required)
# ============================================================

def run_mmd_canary(taus: list[int] = TAU_RANGE) -> None:
    """
    Compute consecutive-pair MMD2 for Cloud A (all nodes) and Cloud B (illicit only).
    Writes rows to DIAG_CSV.

    Guards honored (spec §9):
      §9.1  Cloud A: identical M_A across ALL snapshots.
            Cloud B: identical M_B across ALL qualifying snapshots.
            Per-pair minimum M was the original design but it silently reintroduces the
            size-vs-variance confound identical-M was written to kill: the smallest-M
            transition always produces the largest-variance MMD2 estimate, inflating Z at
            exactly the minimum-N timestamp (τ=43 for illicit nodes).  Fixed M_B removes
            degenerate snapshots rather than including them at arbitrary sub-M.
      §9.3  sigma_A computed once on Cloud A, reused for every pair in both clouds.
            sigma_B computed once on qualifying Cloud B taus (sensitivity check only;
            primary gates use sigma_A throughout).
      §9.4  Global standardizer fitted once on pooled Cloud A features.
      §9.7  Guard band applied identically to null selection in evaluate step.
      §9.9  Non-finite MMD2 aborts immediately.
    """
    print("Loading Cloud A (all nodes)...")
    clouds_A = load_point_clouds(taus)
    print("Loading Cloud B (illicit-labeled nodes only)...")
    clouds_B = load_illicit_clouds(taus)

    valid_A = sorted(t for t in clouds_A if len(clouds_A[t]) > 0)
    valid_B = sorted(t for t in clouds_B if len(clouds_B[t]) > 0)

    # §9.4 Global standardizer — fitted once on Cloud A (all nodes, all tau)
    print("Fitting global standardizer on Cloud A...")
    scaler = fit_global_standardizer({t: clouds_A[t] for t in valid_A})

    # §9.1 Cloud A: identical M_A
    M_A = effective_M({t: clouds_A[t] for t in valid_A})
    print(f"M_A = {M_A}")

    # §9.3 sigma_A — computed on Cloud A subsampled to M_A, seed=0; never recomputed per pair
    rng_sigma = np.random.default_rng(0)
    std_A_M0 = {t: subsample(scaler.transform(clouds_A[t]), M_A, rng_sigma) for t in valid_A}
    sigma = median_heuristic_sigma(std_A_M0)
    _sigma_ref = sigma
    print(f"sigma_A = {sigma:.4f}  (fixed for all pairs and both clouds)")

    rows: list[dict] = []

    # ── Cloud A: all nodes, identical M_A ────────────────────────────────────
    n_a_pairs = sum(1 for i in range(len(valid_A) - 1) if valid_A[i + 1] == valid_A[i] + 1)
    print(f"\nBase: {len(valid_A)} taus, {n_a_pairs} adjacent pairs, {K_SEEDS} seeds")

    for seed in range(K_SEEDS):
        rng = np.random.default_rng(seed)
        std_A_seed = {t: subsample(scaler.transform(clouds_A[t]), M_A, rng) for t in valid_A}
        for i in range(len(valid_A) - 1):
            t, t1 = valid_A[i], valid_A[i + 1]
            if t1 != t + 1:
                continue
            X, Y = std_A_seed[t], std_A_seed[t1]
            assert sigma == _sigma_ref, f"sigma_A drifted at tau={t}"
            mmd2 = mmd2_unbiased(X, Y, sigma)
            assert np.isfinite(mmd2), f"Non-finite MMD2 Base tau={t}→{t1}, seed={seed}"
            rows.append({
                "Sweep": MMD_SWEEP, "Variation": "Base",
                "tau": t, "seed": seed, "shuffle_flag": 0, "M": M_A,
                "beta0_total_pers": np.nan, "beta1_total_pers": np.nan,
                "beta1_max_life": np.nan, "pers_entropy_h0": np.nan, "pers_entropy_h1": np.nan,
                "wasserstein_h0_to_next": np.nan, "wasserstein_h1_to_next": np.nan,
                "wasserstein_combined_to_next": np.nan,
                "mmd2_to_next": mmd2, "Low_Confidence": 0,
            })
        if seed % 2 == 1:
            print(f"  seed {seed + 1}/{K_SEEDS} done", end="\r", flush=True)
    print("\n  Base: complete")

    # ── Cloud B: illicit-only, fixed identical M_B ───────────────────────────
    # Qualification: only taus where N_illicit ≥ M_B.  Degenerate taus (N=2 at τ=46)
    # are excluded rather than included at tiny per-pair M.
    b_n_illicit = {t: len(clouds_B[t]) for t in valid_B}
    qualifying_B = sorted(t for t, n in b_n_illicit.items() if n >= M_B)
    excluded_B   = sorted(t for t, n in b_n_illicit.items() if n <  M_B)
    # §9.1 identical M_B — min of qualifying counts, capped at M_B constant
    M_B_eff = min(M_B, min(b_n_illicit[t] for t in qualifying_B)) if qualifying_B else 0

    # sigma_B — separate kernel scale for Cloud B (sensitivity check; not used in primary gates)
    rng_sigma_B = np.random.default_rng(42)
    std_B_M0 = {
        t: subsample(scaler.transform(clouds_B[t]), M_B_eff, rng_sigma_B)
        for t in qualifying_B
    }
    sigma_B = median_heuristic_sigma(std_B_M0) if qualifying_B else np.nan
    _sigma_B_ref = sigma_B

    n_b_pairs = sum(
        1 for i in range(len(qualifying_B) - 1)
        if qualifying_B[i + 1] == qualifying_B[i] + 1
    )
    print(f"\nClassCond_Illicit: {len(qualifying_B)} qualifying taus (N_illicit ≥ {M_B}), "
          f"M_B={M_B_eff}, sigma_B={sigma_B:.4f}, {n_b_pairs} adjacent pairs, {K_SEEDS} seeds")
    if excluded_B:
        print(f"  Excluded (N_illicit < {M_B}): tau={excluded_B}")

    for seed in range(K_SEEDS):
        rng = np.random.default_rng(seed)
        std_B_seed: dict[int, np.ndarray] = {}
        for t in qualifying_B:
            std_B_seed[t] = subsample(scaler.transform(clouds_B[t]), M_B_eff, rng)

        for i in range(len(qualifying_B) - 1):
            t, t1 = qualifying_B[i], qualifying_B[i + 1]
            if t1 != t + 1:
                continue
            X, Y = std_B_seed[t], std_B_seed[t1]
            # §9.1 identical M_B guard
            assert X.shape[0] == M_B_eff == Y.shape[0], \
                f"Cloud B M mismatch at tau={t}: {X.shape[0]} vs {M_B_eff} vs {Y.shape[0]}"
            assert sigma == _sigma_ref, f"sigma_A drifted at tau={t}"
            assert sigma_B == _sigma_B_ref, f"sigma_B drifted at tau={t}"
            mmd2 = mmd2_unbiased(X, Y, sigma)   # primary uses sigma_A
            assert np.isfinite(mmd2), f"Non-finite MMD2 ClassCond_Illicit tau={t}→{t1}, seed={seed}"
            rows.append({
                "Sweep": MMD_SWEEP, "Variation": "ClassCond_Illicit",
                "tau": t, "seed": seed, "shuffle_flag": 0, "M": M_B_eff,
                "beta0_total_pers": np.nan, "beta1_total_pers": np.nan,
                "beta1_max_life": np.nan, "pers_entropy_h0": np.nan, "pers_entropy_h1": np.nan,
                "wasserstein_h0_to_next": np.nan, "wasserstein_h1_to_next": np.nan,
                "wasserstein_combined_to_next": np.nan,
                "mmd2_to_next": mmd2, "Low_Confidence": 0,   # all qualify by construction
            })
        if seed % 2 == 1:
            print(f"  seed {seed + 1}/{K_SEEDS} done", end="\r", flush=True)
    print("\n  ClassCond_Illicit: complete")

    assert sigma == _sigma_ref, "sigma_A modified during run — integrity violation"
    append_rows(DIAG_CSV, rows, DIAG_COLS)
    print(f"\nWritten {len(rows)} rows to {DIAG_CSV}")


# ============================================================
# EVALUATE: decision matrix + falsification log
# ============================================================

def evaluate_mmd_canary() -> None:
    """
    Read DIAG_CSV, compute per-seed robust Z for T*=(42→43) vs null set,
    emit decision matrix, and append to FALSIFY_CSV.

    H3 (shuffle permutation null with per-permutation max-W) is deferred to the full
    PH arm — it requires the Wasserstein distance matrix from computed diagrams.
    Noted as pending in the falsification log.

    Decision matrix (A = Cloud A / all nodes, B = Cloud B / illicit only):
      A✓ B✓  Real geometric shock at both levels — install ripser/giotto for PH arm.
      A✗ B✓  Shock present in illicit subpop but diluted in the full cloud.
             Base-arm World γ would be a dilution artifact, not evidence of no geometry.
             Illicit-conditioned PH arm is the meaningful target.
      A✗ B✗  Clean World γ — τ=43 is a label-prevalence event only, no geometric
             correlate at either level. Broadcast-bias thesis confirmed geometrically.
             Skip the PH install.
      A✓ B✗  Full-cloud manifold shift without illicit-subpop signal.
             Unknown-population-driven — informative and surprising; investigate.
    """
    if not os.path.exists(DIAG_CSV):
        print(f"No data at {DIAG_CSV}. Run --action run first.")
        return

    df = pd.read_csv(DIAG_CSV)
    df = df[df.Sweep == MMD_SWEEP].copy()
    if len(df) == 0:
        print(f"No rows for sweep '{MMD_SWEEP}' in {DIAG_CSV}")
        return

    # §9.10: thresholds are module-level constants; assert unchanged
    assert H1_Z_MIN == 3.0,      "H1_Z_MIN tampered post-run"
    assert H3_P_MAX == 0.01,     "H3_P_MAX tampered post-run"
    assert M_B == 24,            "M_B tampered post-run"
    assert B_PERM_P_MAX == 0.05, "B_PERM_P_MAX tampered post-run"

    T_STAR = TRANSITION_OF_INTEREST[0]    # 42  (the 42→43 transition row)

    # Re-derive shared artifacts for the permutation test (deterministic, idempotent).
    # Uses TAU_RANGE (1..49) — same as run — so scaler and sigma_A are bit-identical.
    _clouds_A_eval = load_point_clouds(TAU_RANGE)
    _valid_A_eval  = sorted(_clouds_A_eval.keys())
    _scaler_eval   = fit_global_standardizer({t: _clouds_A_eval[t] for t in _valid_A_eval})
    _M_A_eval      = effective_M({t: _clouds_A_eval[t] for t in _valid_A_eval})
    _rng_s = np.random.default_rng(0)
    _sigma_eval = median_heuristic_sigma({
        t: subsample(_scaler_eval.transform(_clouds_A_eval[t]), _M_A_eval, _rng_s)
        for t in _valid_A_eval
    })

    # Also derive sigma_B over all qualifying Cloud B taus (matches run)
    _clouds_B_eval  = load_illicit_clouds(TAU_RANGE)
    _qualifying_B_eval = sorted(
        t for t in _clouds_B_eval if len(_clouds_B_eval[t]) >= M_B
    )
    _rng_sB0 = np.random.default_rng(42)
    _sigma_B_global = median_heuristic_sigma({
        t: subsample(_scaler_eval.transform(_clouds_B_eval[t]), M_B, _rng_sB0)
        for t in _qualifying_B_eval
    }) if _qualifying_B_eval else np.nan

    verdict_fires: dict[str, bool] = {}
    falsify_rows: list[dict] = []

    for variation in ["Base", "ClassCond_Illicit"]:
        sub = df[
            (df.Variation == variation) &
            (df.shuffle_flag == 0) &
            df.mmd2_to_next.notna()
        ]
        if len(sub) == 0:
            print(f"No data for {variation}")
            continue

        # §9.7: guard band applied identically; Low_Confidence=0 by construction for both clouds
        null_mask = (
            sub.tau.apply(lambda t: t not in GUARD and (t + 1) not in GUARD) &
            (sub.Low_Confidence == 0)
        )
        shock_rows = sub[sub.tau == T_STAR]

        if len(shock_rows) == 0:
            print(f"{variation}: no rows for T*=τ={T_STAR}→{T_STAR+1}")
            continue

        seeds = sorted(shock_rows.seed.unique())
        z_per_seed: list[float] = []
        rank_1_count = 0

        for seed in seeds:
            shock_val_rows = shock_rows[shock_rows.seed == seed]["mmd2_to_next"].values
            null_vals = sub[null_mask & (sub.seed == seed)]["mmd2_to_next"].values
            if len(shock_val_rows) == 0 or len(null_vals) < 3:
                continue
            z = robust_z(float(shock_val_rows[0]), null_vals)
            z_per_seed.append(z)

            # Rank of T* among all non-LC transitions in this seed (spec §6 H1 rank-1 gate).
            # Note: in the fixed-M_B design all Cloud B rows have LC=0 so T* is now IN
            # the ranking pool.  In the old per-pair design T* was LC=1 and excluded from
            # its own ranking pool, making rank-1 vacuously impossible — that was a silent
            # spec deviation.  Both clouds now use the full gate.
            all_nonlc = sub[(sub.seed == seed) & (sub.Low_Confidence == 0)]
            if len(all_nonlc) > 0:
                ranked = all_nonlc.sort_values("mmd2_to_next", ascending=False)["tau"].values
                matches = np.where(ranked == T_STAR)[0]
                if len(matches) and int(matches[0]) == 0:
                    rank_1_count += 1

        if not z_per_seed:
            print(f"{variation}: no valid per-seed Z values")
            continue

        median_z    = float(np.median(z_per_seed))
        frac_z_pass = sum(1 for z in z_per_seed if z >= H1_Z_MIN) / len(z_per_seed)
        frac_rank1  = rank_1_count / len(seeds)

        # ── Cloud B permutation test (size-matched, applied at T* only) ──────
        perm_obs, perm_p, sigma_B_eval = np.nan, np.nan, np.nan
        if variation == "ClassCond_Illicit":
            illicit_clouds_eval = load_illicit_clouds([T_STAR, T_STAR + 1])
            if T_STAR in illicit_clouds_eval and (T_STAR + 1) in illicit_clouds_eval:
                X_perm = _scaler_eval.transform(illicit_clouds_eval[T_STAR])
                Y_perm = _scaler_eval.transform(illicit_clouds_eval[T_STAR + 1])
                # Permutation uses full available nodes (all 239 + 24), not M_B-subsampled,
                # to maximize statistical power; repartitions into (239, 24) each permutation.
                perm_obs, perm_p = permutation_test_mmd(X_perm, Y_perm, _sigma_eval,
                                                         n_perm=N_PERM_B, rng_seed=0)
                sigma_B_eval = _sigma_B_global  # over all qualifying B taus, matches run

        # Spec §6 H1 full gate: Z AND seed-fraction AND rank-1.
        # Cloud B additionally requires permutation p-value to be significant.
        # (Conscious deviation from v1 evaluate which omitted rank-1: rank-1 is now
        # restored since T* is LC=0 and participates in the ranking pool.)
        if variation == "ClassCond_Illicit":
            fires = (
                median_z    >= H1_Z_MIN      and
                frac_z_pass >= H1_SEED_FRAC  and
                frac_rank1  >= H1_SEED_FRAC  and
                (not np.isnan(perm_p)) and perm_p <= B_PERM_P_MAX
            )
        else:
            fires = (
                median_z    >= H1_Z_MIN      and
                frac_z_pass >= H1_SEED_FRAC  and
                frac_rank1  >= H1_SEED_FRAC
            )

        n_null_taus = sub[null_mask].tau.nunique()
        print(f"\n{variation}:")
        print(f"  Null set: {n_null_taus} taus, {null_mask.sum()} rows")
        print(f"  Z_mmd per seed: {[f'{z:.2f}' for z in z_per_seed]}")
        print(f"  Median Z_mmd at T*: {median_z:.2f}  (threshold {H1_Z_MIN})")
        print(f"  Seeds with Z ≥ {H1_Z_MIN}: {frac_z_pass:.0%}  (gate {H1_SEED_FRAC:.0%})")
        print(f"  Seeds where T* is rank-1: {frac_rank1:.0%}  (gate {H1_SEED_FRAC:.0%})")
        if variation == "ClassCond_Illicit":
            print(f"  Permutation test (n={N_PERM_B}): observed MMD2={perm_obs:.6f}, "
                  f"p={perm_p:.4f}  (threshold {B_PERM_P_MAX})")
            if not np.isnan(sigma_B_eval):
                print(f"  Sensitivity: sigma_B={sigma_B_eval:.4f} vs sigma_A={_sigma_eval:.4f}"
                      f"  (primary gates use sigma_A)")
            if np.isnan(perm_p):
                print("  NOTE: permutation test skipped (insufficient illicit data at T*)")
        print(f"  FIRES: {fires}")

        # Honesty clause: which transition wins rank-1 most often (Base arm only)
        if variation == "Base":
            top_tau_counts: dict[int, int] = {}
            for seed in seeds:
                all_nonlc = sub[(sub.seed == seed) & (sub.Low_Confidence == 0)]
                if len(all_nonlc):
                    top = int(all_nonlc.sort_values("mmd2_to_next", ascending=False)["tau"].iloc[0])
                    top_tau_counts[top] = top_tau_counts.get(top, 0) + 1
            top_competitor = max(top_tau_counts, key=top_tau_counts.get, default=None)
            if top_competitor is not None and top_competitor != T_STAR:
                print(f"  NOTE: τ={top_competitor}→{top_competitor+1} wins rank-1 in "
                      f"{top_tau_counts[top_competitor]}/{len(seeds)} seeds — "
                      f"all-node arm may respond to prevalence collapses more generally.")

        verdict_fires[variation] = fires
        falsify_rows.append({
            "Sweep":           MMD_SWEEP,
            "hypothesis":      f"MMD_canary_{variation}",
            "statistic_name":  "median_Z_mmd",
            "statistic_value": median_z,
            "threshold":       H1_Z_MIN,
            "seed_fraction":   frac_z_pass,
            "verdict":         "FIRES" if fires else "SILENT",
        })
        if variation == "ClassCond_Illicit" and not np.isnan(perm_p):
            falsify_rows.append({
                "Sweep":           MMD_SWEEP,
                "hypothesis":      "MMD_canary_ClassCond_Illicit_permtest",
                "statistic_name":  "permutation_p",
                "statistic_value": perm_p,
                "threshold":       B_PERM_P_MAX,
                "seed_fraction":   np.nan,
                "verdict":         "SIGNIFICANT" if perm_p <= B_PERM_P_MAX else "NOT_SIGNIFICANT",
            })

    # Decision matrix
    A_fires = verdict_fires.get("Base", False)
    B_fires = verdict_fires.get("ClassCond_Illicit", False)

    if A_fires and B_fires:
        world = "A✓B✓: Real geometric shock at both levels. Install ripser+persim and run PH arm."
    elif not A_fires and B_fires:
        world = ("A✗B✓: Shock in illicit subpop; diluted in full cloud. "
                 "Base-arm γ is likely a dilution artifact. "
                 "Illicit-conditioned PH arm is the meaningful target.")
    elif not A_fires and not B_fires:
        world = ("A✗B✗: Clean World γ. τ=43 is a label-prevalence event only — "
                 "no geometric correlate at either level. Skip PH install. "
                 "Broadcast-bias thesis confirmed geometrically.")
    else:
        world = ("A✓B✗: Full-cloud manifold shift without illicit-subpop signal. "
                 "Unknown-population-driven shift. Informative and surprising.")

    print(f"\n{'='*60}")
    print("DECISION MATRIX")
    print(world)
    print(f"{'='*60}")
    print("\nH3 (shuffle permutation null with per-permutation max-W): DEFERRED.")
    print("Requires Wasserstein distance matrix from PH diagrams — implement in PH arm.")

    falsify_rows.append({
        "Sweep":           MMD_SWEEP,
        "hypothesis":      "decision_matrix",
        "statistic_name":  "world",
        "statistic_value": np.nan,
        "threshold":       np.nan,
        "seed_fraction":   np.nan,
        "verdict":         world,
    })
    falsify_rows.append({
        "Sweep":           MMD_SWEEP,
        "hypothesis":      "H3",
        "statistic_name":  "shuffle_null",
        "statistic_value": np.nan,
        "threshold":       H3_P_MAX,
        "seed_fraction":   np.nan,
        "verdict":         "DEFERRED: needs PH Wasserstein matrix",
    })

    append_rows(FALSIFY_CSV, falsify_rows, FALSIFY_COLS)
    print(f"\nFalsification log written to {FALSIFY_CSV}")


# ============================================================
# FULL PH ARM STUBS (wired after MMD canary shows a target)
# ============================================================

def pooled_max_edge_length(std_clouds_M: dict[int, np.ndarray]) -> float:
    """ME fixed across ALL snapshots = ME_PERCENTILE pct of pooled pairwise dists."""
    sample = np.vstack([std_clouds_M[t] for t in sorted(std_clouds_M)])
    if sample.shape[0] > 2000:
        r = np.random.default_rng(0)
        sample = sample[r.choice(sample.shape[0], 2000, replace=False)]
    d = pairwise_distances(sample)
    return float(np.percentile(d[np.triu_indices_from(d, k=1)], ME_PERCENTILE))


def compute_diagrams(clouds_M: dict[int, np.ndarray], me: float, metric: str):
    """VR persistence diagrams. Requires giotto-tda or ripser. Wired in PH arm."""
    # TODO[ph-arm]: call giotto VietorisRipsPersistence or ripser.ripser here.
    # Identical M and ME must be asserted before calling (spec §9.1, §9.2).
    raise NotImplementedError("PH arm: install ripser or giotto-tda first")


def consecutive_wasserstein(diagrams, taus: list[int]) -> dict[int, dict]:
    """Per-dim Wasserstein between consecutive diagrams. PH arm only."""
    # TODO[ph-arm]: implement using persim.wasserstein (ripser path) or
    # gtda.diagrams.PairwiseDistance (giotto path). The full nxn matrix is computed
    # once per seed so the shuffle null (H3) reads off-diagonal entries without
    # recomputing Wasserstein — pass the precomputed matrix to the shuffle loop.
    raise NotImplementedError("PH arm: implement after library choice is resolved")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="TDA diagnostic of the tau=43 regime shift")
    ap.add_argument(
        "--action", choices=["smoke", "run", "evaluate"], required=True,
        help="smoke: validate plumbing; run: MMD canary; evaluate: decision matrix",
    )
    ap.add_argument("--taus", default="1-49",
                    help="Tau range, e.g. '1-49' or '35-49' (default: 1-49)")
    args = ap.parse_args()

    if args.action == "smoke":
        smoke_test()
        return

    # Parse tau range
    lo, hi = (int(x) for x in args.taus.split("-"))
    taus = list(range(lo, hi + 1))

    if args.action == "run":
        run_mmd_canary(taus)
    elif args.action == "evaluate":
        evaluate_mmd_canary()


if __name__ == "__main__":
    main()
