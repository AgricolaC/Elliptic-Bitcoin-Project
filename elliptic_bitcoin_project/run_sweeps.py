"""
run_sweeps.py — Ablation sweep runner.

W4 FIX: Expanded ablation matrix that isolates each mechanism independently.
W5 FIX: All result dicts use the standardized key schema:
         {"Sweep", "Static OOT F1", "Static OOT PR-AUC",
          "Walk-Forward Mean F1", "Walk-Forward Mean PR-AUC"}
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import warnings
import numpy as np
import pandas as pd
import joblib
import re
from sklearn.metrics import f1_score, average_precision_score

from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, walk_forward_validation

warnings.filterwarnings("ignore", category=UserWarning)

_RESULT_KEYS = (
    "Sweep",
    "Static Time (s)",
    "Static Mem (MB)",
    "Static OOT F1",
    "Static OOT PR-AUC",
    "WF Time (s)",
    "WF Mem (MB)",
    "Walk-Forward Mean F1",
    "Walk-Forward Mean PR-AUC",
)

import tracemalloc
import time
from contextlib import contextmanager

@contextmanager
def profile_resources():
    tracemalloc.start()
    start_t = time.perf_counter()
    metrics = {}
    try:
        yield metrics
    finally:
        end_t = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["time"] = end_t - start_t
        metrics["peak_mem"] = peak / (1024 * 1024)


def _make_result(
    sweep: str,
    static_time: float | str,
    static_mem: float | str,
    static_f1: float | str,
    static_prauc: float | str,
    wf_time: float | str,
    wf_mem: float | str,
    wf_f1: float | str,
    wf_prauc: float | str,
) -> dict:
    """
    Construct a result dict with the standardized key schema (W5 fix).
    Raises AssertionError if any key would be missing.
    """
    result = {
        "Sweep":                    sweep,
        "Static Time (s)":          static_time,
        "Static Mem (MB)":          static_mem,
        "Static OOT F1":            static_f1,
        "Static OOT PR-AUC":        static_prauc,
        "WF Time (s)":              wf_time,
        "WF Mem (MB)":              wf_mem,
        "Walk-Forward Mean F1":     wf_f1,
        "Walk-Forward Mean PR-AUC": wf_prauc,
    }
    # SHAPE GUARD: verify key completeness on every call
    assert set(result.keys()) == set(_RESULT_KEYS), \
        f"Result key schema violation: {set(result.keys())} != {set(_RESULT_KEYS)}"
    return result


def run_single_sweep(
    name: str,
    cfg: Config,
    df: pd.DataFrame,
    df_edge: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Run one full sweep (static OOT + walk-forward) and return a standardized result dict.

    W5 FIX: Returns _make_result(...) which enforces the canonical key schema.
    W6 FIX: dm.setup() now runs propagation internally; no external assignment needed.
    W7 FIX: Passes sweep name to walk_forward_validation for unique plot filenames.
    W8 FIX: Class weights are computed inside walk_forward_validation per tau.
    """
    set_global_seeds(cfg.seed)

    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    # W6: sgc_input_dim and 'prop' keys are now set inside setup() — no manual step.

    Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
    Xte_g, yte_g = stack_prop(dm, list(cfg.test_steps))

    # Static OOT class weights (used only for the static evaluation head)
    valid_ytr = ytr_g[ytr_g != -1]
    counts    = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
    cls_w     = (counts.sum() / (2.0 * counts)).to(DEVICE)

    with profile_resources() as stat_res:
        model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
        model.eval()
        with torch.no_grad():
            m      = (yte_g != -1)
            scores = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()

        y_true       = yte_g[m].numpy()
        static_f1    = f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)
        static_prauc = average_precision_score(y_true, scores)

    with profile_resources() as wf_res:
        wf_f1, wf_prauc, wf_records = walk_forward_validation(
            dm, cfg, DEVICE, sweep_name=name, return_records=True
        )

    safe_name = re.sub(r"[^\w\-]", "_", name)
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Dump static OOT model, DM, and Walk-Forward records
    joblib.dump(dm, os.path.join(model_dir, f"{safe_name}_dm.pkl"))
    joblib.dump(cfg, os.path.join(model_dir, f"{safe_name}_cfg.pkl"))
    torch.save(model.state_dict(), os.path.join(model_dir, f"{safe_name}_model.pt"))
    joblib.dump(wf_records, os.path.join(model_dir, f"{safe_name}_wf_records.pkl"))

    return _make_result(
        sweep=name,
        static_time=round(stat_res.get("time", 0.0), 3),
        static_mem=round(stat_res.get("peak_mem", 0.0), 2),
        static_f1=round(static_f1, 3),
        static_prauc=round(static_prauc, 3),
        wf_time=round(wf_res.get("time", 0.0), 3),
        wf_mem=round(wf_res.get("peak_mem", 0.0), 2),
        wf_f1=round(wf_f1, 3),
        wf_prauc=round(wf_prauc, 3),
    )


def run_static_only_sweep(
    name: str,
    cfg: Config,
    df: pd.DataFrame,
    df_edge: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """Run a fast static OOT sweep (skips walk-forward validation)."""
    set_global_seeds(cfg.seed)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
    Xte_g, yte_g = stack_prop(dm, list(cfg.test_steps))

    valid_ytr = ytr_g[ytr_g != -1]
    counts    = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
    cls_w     = (counts.sum() / (2.0 * counts)).to(DEVICE)

    with profile_resources() as stat_res:
        model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
        model.eval()
        with torch.no_grad():
            m      = (yte_g != -1)
            scores = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()

        y_true       = yte_g[m].numpy()
        static_f1    = f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)
        static_prauc = average_precision_score(y_true, scores)
    
    # Save the static-only model + dm + cfg for potential later analysis
    safe_name = re.sub(r"[^\w\-]", "_", name)
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(dm, os.path.join(model_dir, f"{safe_name}_dm.pkl"))
    joblib.dump(cfg, os.path.join(model_dir, f"{safe_name}_cfg.pkl"))
    torch.save(model.state_dict(), os.path.join(model_dir, f"{safe_name}_model.pt"))

    return _make_result(
        sweep=name,
        static_time=round(stat_res.get("time", 0.0), 3),
        static_mem=round(stat_res.get("peak_mem", 0.0), 2),
        static_f1=round(static_f1, 3),
        static_prauc=round(static_prauc, 3),
        wf_time="N/A",
        wf_mem="N/A",
        wf_f1="N/A",
        wf_prauc="N/A",
    )


def walk_forward_baseline(dm: EllipticDataModule, cfg: Config, model_cls, **model_kwargs) -> tuple:
    """Walk-forward evaluation for scikit-learn/XGBoost tabular models."""
    wf_f1s = []
    wf_praucs = []
    
    for tau in cfg.test_steps:
        # Train on [1, tau-1]
        Xs_tr, ys_tr = [], []
        for t in range(1, tau):
            g = dm.graphs[t]
            m = g["labeled_mask"].numpy()
            if m.sum() > 0:
                Xs_tr.append(g["x"].numpy()[:, :166][m])
                ys_tr.append(g["y"].numpy()[m])
                
        if len(Xs_tr) == 0:
            continue
            
        Xtr = np.concatenate(Xs_tr)
        ytr = np.concatenate(ys_tr)
        
        if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
            continue
            
        # Test on tau
        g_tau = dm.graphs[tau]
        m_tau = g_tau["labeled_mask"].numpy()
        if m_tau.sum() == 0:
            continue
            
        Xte = g_tau["x"].numpy()[:, :166][m_tau]
        yte = g_tau["y"].numpy()[m_tau]
        
        # Train & Predict
        model = model_cls(**model_kwargs).fit(Xtr, ytr)
        s_pred = model.predict_proba(Xte)[:, 1]
        y_pred = (s_pred >= 0.5).astype(int)
        
        wf_f1s.append(f1_score(yte, y_pred, pos_label=1, zero_division=0))
        wf_praucs.append(average_precision_score(yte, s_pred))
        
    return np.mean(wf_f1s), np.mean(wf_praucs)


def main():
    from data.load_dataset import download_and_load_data
    from analysis.eda import plot_temporal_distribution

    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    print("\n--- Phase 1: EDA ---")
    cfg_default = Config()
    plot_temporal_distribution(df, cfg_default)

    results = []

    # ── Baselines (tabular, no GNN) ────────────────────────────────────────────
    print("\n--- Baseline Tabular Models ---")
    try:
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        dm_base = EllipticDataModule(df, df_edge, feature_cols, cfg_default)
        dm_base.setup()

        Xs_tr, ys_tr = [], []
        for t in cfg_default.train_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_tr.append(g["x"].numpy()[:, :166][m])
            ys_tr.append(g["y"].numpy()[m])
        Xtr_b, ytr_b = np.concatenate(Xs_tr), np.concatenate(ys_tr)

        Xs_te, ys_te = [], []
        for t in cfg_default.test_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_te.append(g["x"].numpy()[:, :166][m])
            ys_te.append(g["y"].numpy()[m])
        Xte_b, yte_b = np.concatenate(Xs_te), np.concatenate(ys_te)

        spw = (ytr_b == 0).sum() / max((ytr_b == 1).sum(), 1)
        with profile_resources() as stat_xgb:
            xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 scale_pos_weight=spw, eval_metric="aucpr",
                                 random_state=cfg_default.seed, n_jobs=1).fit(Xtr_b, ytr_b)
            s_xgb = xgb.predict_proba(Xte_b)[:, 1]
            static_xgb_f1 = f1_score(yte_b, (s_xgb >= 0.5).astype(int), pos_label=1)
            static_xgb_prauc = average_precision_score(yte_b, s_xgb)
            
        with profile_resources() as wf_xgb:
            wf_xgb_f1, wf_xgb_prauc = walk_forward_baseline(
                dm_base, cfg_default, XGBClassifier,
                n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw, eval_metric="aucpr", random_state=cfg_default.seed, n_jobs=1
            )
        
        os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
        joblib.dump(xgb, os.path.join(OUTPUT_DIR, "models", "xgb_baseline.pkl"))

        with profile_resources() as stat_rf:
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        n_jobs=1, random_state=cfg_default.seed).fit(Xtr_b, ytr_b)
            s_rf = rf.predict_proba(Xte_b)[:, 1]
            static_rf_f1 = f1_score(yte_b, (s_rf >= 0.5).astype(int), pos_label=1)
            static_rf_prauc = average_precision_score(yte_b, s_rf)
            
        with profile_resources() as wf_rf:
            wf_rf_f1, wf_rf_prauc = walk_forward_baseline(
                dm_base, cfg_default, RandomForestClassifier,
                n_estimators=200, class_weight="balanced", n_jobs=1, random_state=cfg_default.seed
            )

        results.append(_make_result(
            "Baseline: XGBoost (166)",
            static_time=round(stat_xgb.get("time", 0.0), 3),
            static_mem=round(stat_xgb.get("peak_mem", 0.0), 2),
            static_f1=round(static_xgb_f1, 3),
            static_prauc=round(static_xgb_prauc, 3),
            wf_time=round(wf_xgb.get("time", 0.0), 3),
            wf_mem=round(wf_xgb.get("peak_mem", 0.0), 2),
            wf_f1=round(wf_xgb_f1, 3),
            wf_prauc=round(wf_xgb_prauc, 3),
        ))
        results.append(_make_result(
            "Baseline: RandomForest (166)",
            static_time=round(stat_rf.get("time", 0.0), 3),
            static_mem=round(stat_rf.get("peak_mem", 0.0), 2),
            static_f1=round(static_rf_f1, 3),
            static_prauc=round(static_rf_prauc, 3),
            wf_time=round(wf_rf.get("time", 0.0), 3),
            wf_mem=round(wf_rf.get("peak_mem", 0.0), 2),
            wf_f1=round(wf_rf_f1, 3),
            wf_prauc=round(wf_rf_prauc, 3),
        ))
    except Exception as e:
        print(f"  Baselines skipped: {e}")

    # ── W4 FIX: Expanded ablation matrix ──────────────────────────────────────
    # Each sweep toggles exactly ONE mechanism relative to the previous row,
    # enabling unambiguous attribution of gain.
    sweeps = [
        # name                          use_mlp  use_ms   use_topo  use_recon  use_focal
        ("Sweep 1: SGC (baseline)",
         Config(use_mlp_head=False, use_multiscale_prop=False,
                use_topology=False)),

        ("Sweep 2: + MLP Head",
         Config(use_mlp_head=True,  use_multiscale_prop=False,
                use_topology=False)),

        ("Sweep 3: + Multiscale Prop",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=False)),

        ("Sweep 4: + Topology Features",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=True))
    ]

    for name, cfg in sweeps:
        print(f"\n{'='*55}\nRunning: {name}\n{'='*55}")
        res = run_single_sweep(name, cfg, df, df_edge, feature_cols)
        results.append(res)
        
        # Incremental save
        pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
        print(f"--> {res}\n")

    print("\n--- K Ablation (Static Only) ---")
    k_sweeps = [
        ("Sweep K=1 (Static)", Config(sgc_k=1, use_multiscale_prop=True, use_topology=True, use_mlp_head=True)),
        ("Sweep K=2 (Static)", Config(sgc_k=2, use_multiscale_prop=True, use_topology=True, use_mlp_head=True)),
        ("Sweep K=3 (Static)", Config(sgc_k=3, use_multiscale_prop=True, use_topology=True, use_mlp_head=True)),
    ]
    for name, cfg in k_sweeps:
        print(f"Running: {name}")
        res = run_static_only_sweep(name, cfg, df, df_edge, feature_cols)
        results.append(res)
        pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
        print(f"--> {res}\n")


    # ── Advanced modules ───────────────────────────────────────────────────────
    cfg_full = sweeps[-1][1]
    dm_adv   = EllipticDataModule(df, df_edge, feature_cols, cfg_full)
    dm_adv.setup()

    try:
        from models.drift_adaptation import explicit_drift_adaptation
        with profile_resources() as wf_drift:
            res_drift = explicit_drift_adaptation(dm_adv, cfg_full)
            
        results.append(_make_result(
            res_drift["Sweep"],
            static_time="N/A",
            static_mem="N/A",
            static_f1=res_drift.get("Static OOT F1", "N/A"),
            static_prauc=res_drift.get("Static OOT PR-AUC", "N/A"),
            wf_time=round(wf_drift.get("time", 0.0), 3),
            wf_mem=round(wf_drift.get("peak_mem", 0.0), 2),
            wf_f1=res_drift.get("Walk-Forward Mean F1", "N/A"),
            wf_prauc=res_drift.get("Walk-Forward Mean PR-AUC", "N/A"),
        ))
    except Exception as e:
        print(f"  Drift adaptation skipped: {e}")



    # ── Persist results ────────────────────────────────────────────────────────
    df_res   = pd.DataFrame(results, columns=list(_RESULT_KEYS))
    out_file = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    df_res.to_csv(out_file, index=False)
    print(f"\nResults saved to {out_file}")

    # Verify no NaN from key mismatch (W5 guard)
    nan_cols = df_res.columns[df_res.isna().any()].tolist()
    assert not nan_cols, (
        f"W5 GUARD: NaN columns found in sweep_results.csv: {nan_cols}. "
        f"This indicates a key schema mismatch — check _make_result usage."
    )

    print("\n--- FINAL ABLATION RESULTS ---")
    for r in results:
        print(
            f"{r['Sweep']:35s} | "
            f"Stat [Time:{str(r['Static Time (s)']):>5s}s, Mem:{str(r['Static Mem (MB)']):>5s}MB] F1={str(r['Static OOT F1']):<5s} | "
            f"WF [Time:{str(r['WF Time (s)']):>5s}s, Mem:{str(r['WF Mem (MB)']):>5s}MB] F1={str(r['Walk-Forward Mean F1']):<5s}"
        )


if __name__ == "__main__":
    main()
