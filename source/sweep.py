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
    "Seed",
    "Variation",
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
    seed: int,
    variation: str,
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
        "Seed":                     seed,
        "Variation":                variation,
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
    window: int = None,
    variation: str = "Base",
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
            dm, cfg, DEVICE, sweep_name=name, return_records=True, window=window,
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
        seed=cfg.seed,
        variation=variation,
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
    variation: str = "Base",
) -> dict:
    """Run a fast static OOT sweep (skips walk-forward validation)."""
    set_global_seeds(cfg.seed)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
    Xte_g, yte_g = stack_prop(dm, list(cfg.test_steps))

    if variation == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=cfg.seed)
        Xtr_g = torch.tensor(pca.fit_transform(Xtr_g.numpy()), dtype=torch.float32)
        Xte_g = torch.tensor(pca.transform(Xte_g.numpy()), dtype=torch.float32)
    elif variation == "RF_Pruned":
        from sklearn.ensemble import RandomForestClassifier
        m_tr = (ytr_g != -1)
        rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=cfg.seed)
        rf.fit(Xtr_g[m_tr].numpy(), ytr_g[m_tr].numpy())
        mask = rf.feature_importances_ > 0.000
        if mask.sum() > 0:
            Xtr_g = Xtr_g[:, mask]
            Xte_g = Xte_g[:, mask]

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
        seed=cfg.seed,
        variation=variation,
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


def walk_forward_baseline(dm: EllipticDataModule, cfg: Config, model_cls, window: int = None, **model_kwargs) -> tuple:
    """Walk-forward evaluation for scikit-learn/XGBoost tabular models."""
    y_true_all = []
    y_pred_all = []
    y_score_all = []
    
    for tau in cfg.test_steps:
        # Train on [1, tau-1]
        Xs_tr, ys_tr = [], []
        t_start = max(1, tau - window) if window else 1
        for t in range(t_start, tau):
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
        s = model.predict_proba(Xte)[:, 1]
        y_pred = (s >= 0.5).astype(int)
        
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        y_score_all.append(s)
            
    from evaluation.validation import _aggregate_walk_forward
    pooled_f1, pooled_prauc, macro_f1, macro_prauc, pooled_pak = _aggregate_walk_forward(y_true_all, y_pred_all, y_score_all)
    return pooled_f1, pooled_prauc


def walk_forward_isoforest(dm, cfg):
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import f1_score, average_precision_score
    y_true_all, y_pred_all, y_score_all = [], [], []
    for tau in cfg.test_steps:
        train_block = [t for t in range(min(dm.graphs), tau) if t in dm.graphs]
        if not train_block:
            continue
        Xtr = np.concatenate([dm.graphs[t]["x"].numpy()[:, :166] for t in train_block])
        g = dm.graphs[tau]
        m = g["labeled_mask"].numpy()
        Xte = g["x"].numpy()[:, :166][m]
        yte = g["y"].numpy()[m]
        if len(yte) == 0 or len(np.unique(yte)) < 2:
            continue
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=cfg.seed, n_jobs=1)
        iso.fit(Xtr)
        scores = -iso.score_samples(Xte)  # higher = more anomalous
        actual_rate = (yte == 1).mean()
        y_pred = (scores >= np.percentile(scores, (1 - actual_rate) * 100)).astype(int)
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        y_score_all.append(scores)
    from evaluation.validation import _aggregate_walk_forward
    pooled_f1, pooled_prauc, _, _, _ = _aggregate_walk_forward(y_true_all, y_pred_all, y_score_all)
    return pooled_f1, pooled_prauc

def walk_forward_ae(dm, cfg, device):
    import numpy as np
    import torch
    from evaluation.validation import fit_autoencoder, _aggregate_walk_forward
    from sklearn.metrics import f1_score, average_precision_score
    y_true_all, y_pred_all, y_score_all = [], [], []
    for tau in cfg.test_steps:
        train_block = [t for t in range(min(dm.graphs), tau) if t in dm.graphs]
        if not train_block:
            continue
        Xtr_np = np.concatenate([dm.graphs[t]["x"].numpy()[:, :166] for t in train_block])
        Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
        
        ae = fit_autoencoder(Xtr, 166, cfg, device, epochs=60, lr=1e-3)
        
        g = dm.graphs[tau]
        m = g["labeled_mask"].numpy()
        Xte_np = g["x"].numpy()[:, :166][m]
        yte = g["y"].numpy()[m]
        if len(yte) == 0 or len(np.unique(yte)) < 2:
            continue
            
        Xte = torch.tensor(Xte_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            x_hat = ae(Xte)
            scores = ((Xte - x_hat) ** 2).mean(dim=1).cpu().numpy()
            
        actual_rate = (yte == 1).mean()
        thresh_pct = (1 - actual_rate) * 100
        y_pred = (scores >= np.percentile(scores, thresh_pct)).astype(int)
        
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        y_score_all.append(scores)
        
    pooled_f1, pooled_prauc, _, _, _ = _aggregate_walk_forward(y_true_all, y_pred_all, y_score_all)
    return pooled_f1, pooled_prauc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "mega"])
    args = parser.parse_args()

    from data.load_dataset import download_and_load_data

    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg_default = Config()

    results = []
    completed_sweeps = set()
    out_file = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    if os.path.exists(out_file):
        try:
            df_res = pd.read_csv(out_file, keep_default_na=False)
            completed_sweeps = set(df_res["Sweep"].tolist())
            results = df_res.to_dict('records')
            print(f"Loaded {len(completed_sweeps)} completed sweeps from {out_file}")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    # ── Baselines (tabular, no GNN) ────────────────────────────────────────────
    print("\n--- Baseline Tabular Models ---")
    try:
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        cfg_tabular = Config(use_graph_structural=False, sgc_k=0, use_multiscale_prop=False, seed=cfg_default.seed)
        dm_base = EllipticDataModule(df, df_edge, feature_cols, cfg_tabular)
        dm_base.setup()

        Xs_tr, ys_tr = [], []
        for t in cfg_tabular.train_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_tr.append(g["x"].numpy()[:, :166][m])
            ys_tr.append(g["y"].numpy()[m])
        Xtr_b, ytr_b = np.concatenate(Xs_tr), np.concatenate(ys_tr)

        Xs_te, ys_te = [], []
        for t in cfg_tabular.test_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_te.append(g["x"].numpy()[:, :166][m])
            ys_te.append(g["y"].numpy()[m])
        Xte_b, yte_b = np.concatenate(Xs_te), np.concatenate(ys_te)

        if "Baseline: IsolationForest (166)" not in completed_sweeps:
            from sklearn.ensemble import IsolationForest
            with profile_resources() as stat_iso:
                Xtr_iso = np.concatenate([dm_base.graphs[t]["x"].numpy()[:, :166] for t in cfg_tabular.train_steps])
                Xte_iso = Xte_b
                iso = IsolationForest(n_estimators=100, contamination='auto', random_state=cfg_tabular.seed, n_jobs=1)
                iso.fit(Xtr_iso)
                scores = -iso.score_samples(Xte_iso)
                actual_illicit_rate = (yte_b == 1).mean()
                thresh_pct = (1 - actual_illicit_rate) * 100
                static_iso_f1 = f1_score(yte_b, (scores >= np.percentile(scores, thresh_pct)).astype(int), pos_label=1, zero_division=0)
                static_iso_prauc = average_precision_score(yte_b, scores)
                
            with profile_resources() as wf_iso:
                wf_iso_f1, wf_iso_prauc = walk_forward_isoforest(dm_base, cfg_tabular)
                
            results.append(_make_result(
                seed=cfg_tabular.seed,
                variation="Base",
                sweep="Baseline: IsolationForest (166)",
                static_time=round(stat_iso.get("time", 0.0), 3),
                static_mem=round(stat_iso.get("peak_mem", 0.0), 2),
                static_f1=round(static_iso_f1, 3),
                static_prauc=round(static_iso_prauc, 3),
                wf_time=round(wf_iso.get("time", 0.0), 3),
                wf_mem=round(wf_iso.get("peak_mem", 0.0), 2),
                wf_f1=round(wf_iso_f1, 3),
                wf_prauc=round(wf_iso_prauc, 3),
            ))
            pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
        else:
            print("Already completed Baseline: IsolationForest (166), skipping.")



        if "Baseline: XGBoost (166)" not in completed_sweeps:
            spw = (ytr_b == 0).sum() / max((ytr_b == 1).sum(), 1)
            with profile_resources() as stat_xgb:
                xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     scale_pos_weight=spw, eval_metric="aucpr",
                                     random_state=cfg_tabular.seed, n_jobs=1).fit(Xtr_b, ytr_b)
                s_xgb = xgb.predict_proba(Xte_b)[:, 1]
                static_xgb_f1 = f1_score(yte_b, (s_xgb >= 0.5).astype(int), pos_label=1)
                static_xgb_prauc = average_precision_score(yte_b, s_xgb)
                
            with profile_resources() as wf_xgb:
                wf_xgb_f1, wf_xgb_prauc = walk_forward_baseline(
                    dm_base, cfg_tabular, XGBClassifier, window=None,
                    n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw, eval_metric="aucpr", random_state=cfg_tabular.seed, n_jobs=1
                )
            
            os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
            joblib.dump(xgb, os.path.join(OUTPUT_DIR, "models", "xgb_baseline.pkl"))

            results.append(_make_result(
                seed=cfg_tabular.seed,
                variation="Base",
                sweep="Baseline: XGBoost (166)",
                static_time=round(stat_xgb.get("time", 0.0), 3),
                static_mem=round(stat_xgb.get("peak_mem", 0.0), 2),
                static_f1=round(static_xgb_f1, 3),
                static_prauc=round(static_xgb_prauc, 3),
                wf_time=round(wf_xgb.get("time", 0.0), 3),
                wf_mem=round(wf_xgb.get("peak_mem", 0.0), 2),
                wf_f1=round(wf_xgb_f1, 3),
                wf_prauc=round(wf_xgb_prauc, 3),
            ))
            pd.DataFrame(results).to_csv(out_file, index=False)
        else:
            print("Already completed Baseline: XGBoost (166), skipping.")

        if "Baseline: RandomForest (166)" not in completed_sweeps:
            with profile_resources() as stat_rf:
                rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            n_jobs=1, random_state=cfg_tabular.seed).fit(Xtr_b, ytr_b)
                s_rf = rf.predict_proba(Xte_b)[:, 1]
                static_rf_f1 = f1_score(yte_b, (s_rf >= 0.5).astype(int), pos_label=1)
                static_rf_prauc = average_precision_score(yte_b, s_rf)
                
            with profile_resources() as wf_rf:
                wf_rf_f1, wf_rf_prauc = walk_forward_baseline(
                    dm_base, cfg_tabular, RandomForestClassifier, window=None,
                    n_estimators=200, class_weight="balanced", n_jobs=1, random_state=cfg_tabular.seed
                )

            results.append(_make_result(
                seed=cfg_tabular.seed,
                variation="Base",
                sweep="Baseline: RandomForest (166)",
                static_time=round(stat_rf.get("time", 0.0), 3),
                static_mem=round(stat_rf.get("peak_mem", 0.0), 2),
                static_f1=round(static_rf_f1, 3),
                static_prauc=round(static_rf_prauc, 3),
                wf_time=round(wf_rf.get("time", 0.0), 3),
                wf_mem=round(wf_rf.get("peak_mem", 0.0), 2),
                wf_f1=round(wf_rf_f1, 3),
                wf_prauc=round(wf_rf_prauc, 3),
            ))
            pd.DataFrame(results).to_csv(out_file, index=False)
        else:
            print("Already completed Baseline: RandomForest (166), skipping.")
    except Exception as e:
        print(f"  Baselines skipped: {e}")

    # ── W4 FIX: Expanded ablation matrix ──────────────────────────────────────
    # Each sweep toggles exactly ONE mechanism relative to the previous row,
    # enabling unambiguous attribution of gain.
    sweeps = [
        # name                          use_mlp  use_ms   use_topo  use_recon  use_focal
        ("Sweep 1: SGC (baseline)",
         Config(use_mlp_head=False, use_multiscale_prop=False,
                use_graph_structural=False)),

        ("Sweep 2: + MLP Head",
         Config(use_mlp_head=True,  use_multiscale_prop=False,
                use_graph_structural=False)),

        ("Sweep 3: + Multiscale Prop",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_graph_structural=False)),

        ("Sweep 4: + Graph Structure Features (PageRank + Clustering Coeff.)",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_graph_structural=True)),

        ("Sweep 5: + Directional Channels",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_graph_structural=True, use_directional_prop=True))
    ]

    seeds = [42, 43, 44] if args.mode == "mega" else [42]
    variations = ["Base", "PCA", "RF_Pruned"] if args.mode == "mega" else ["Base"]

    for seed in seeds:
        for name, cfg in sweeps:
            cfg.seed = seed
            for var in variations:
                sweep_key = f"{name} (Seed {seed}, Var {var})" if args.mode == "mega" else name
                print(f"\n{'='*55}\nRunning: {sweep_key}\n{'='*55}")
                
                if sweep_key in completed_sweeps:
                    print(f"Already completed {sweep_key}, skipping.")
                    continue
                    
                if args.mode == "standard":
                    res = run_single_sweep(name, cfg, df, df_edge, feature_cols, variation=var)
                else:
                    res = run_static_only_sweep(name, cfg, df, df_edge, feature_cols, variation=var)
                
                res["Sweep"] = sweep_key
                results.append(res)
                
                pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
                print(f"--> {res}\n")

    print("\n--- K Ablation (Static Only) ---")
    k_sweeps = [
        ("Sweep K=1 (Static)", Config(sgc_k=1, use_multiscale_prop=True, use_graph_structural=True, use_mlp_head=True)),
        ("Sweep K=2 (Static)", Config(sgc_k=2, use_multiscale_prop=True, use_graph_structural=True, use_mlp_head=True)),
        ("Sweep K=3 (Static)", Config(sgc_k=3, use_multiscale_prop=True, use_graph_structural=True, use_mlp_head=True)),
    ]
    for name, cfg in k_sweeps:
        print(f"Running: {name}")
        if name in completed_sweeps:
            print(f"Already completed {name}, skipping.")
            continue
        res = run_static_only_sweep(name, cfg, df, df_edge, feature_cols)
        results.append(res)
        pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
        print(f"--> {res}\n")


    # ── Walk-Forward on Best SGC Configuration ────────────────────────────────
    print("\n--- Walk-Forward Validation (Best SGC Sweep) ---")
    best_f1 = -1.0
    best_sweep_name = None
    for r in results:
        if isinstance(r.get("Sweep"), str) and r["Sweep"].startswith("Sweep ") and "K=" not in r["Sweep"]:
            f1_val = r.get("Static OOT F1", 0.0)
            if pd.notna(f1_val) and isinstance(f1_val, (int, float)) and f1_val > best_f1:
                best_f1 = f1_val
                best_sweep_name = r["Sweep"]

    if best_sweep_name:
        wf_name = f"Best WF: {best_sweep_name}"
        if wf_name not in completed_sweeps:
            print(f"\nWinning Configuration: {best_sweep_name} (Static F1: {best_f1:.3f})")
            best_cfg = next((cfg for name, cfg in sweeps if name == best_sweep_name), None)
            
            if best_cfg is not None:
                with profile_resources() as wf_stat:
                    dm_best = EllipticDataModule(df, df_edge, feature_cols, best_cfg)
                    dm_best.setup()
                    wf_f1, wf_prauc = walk_forward_validation(dm_best, best_cfg, DEVICE, sweep_name=wf_name)
                    
                wf_res = _make_result(
                    seed=best_cfg.seed,
                    variation="Base",
                    sweep=wf_name,
                    static_time="N/A",
                    static_mem="N/A",
                    static_f1="N/A",
                    static_prauc="N/A",
                    wf_time=round(wf_stat.get("time", 0.0), 3),
                    wf_mem=round(wf_stat.get("peak_mem", 0.0), 2),
                    wf_f1=round(wf_f1, 3),
                    wf_prauc=round(wf_prauc, 3),
                )
                results.append(wf_res)
                pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
                print(f"--> {wf_res}\n")
        else:
            print(f"Already completed {wf_name}, skipping.")


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
