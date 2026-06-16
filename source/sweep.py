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
from evaluation.temporal_validation import walk_forward_lstm_conditioned, walk_forward_ema_conditioned

warnings.filterwarnings("ignore", category=UserWarning)

_RESULT_KEYS = (
    "Seed",
    "Variation",
    "Sweep",
    "Feature Set",
    "Threshold",
    "Static Time (s)",
    "Static Mem (MB)",
    "Static Val F1",
    "Static Val PR-AUC",
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
    feature_set: str = "N/A",
    threshold: str = "0.5",
    val_f1: float | str = "N/A",
    val_prauc: float | str = "N/A",
) -> dict:
    """
    Construct a result dict with the standardized key schema (W5 fix).
    P1-B: Now includes Feature Set, Threshold, Static Val F1/PR-AUC columns.
    Raises AssertionError if any key would be missing.
    """
    result = {
        "Seed":                     seed,
        "Variation":                variation,
        "Sweep":                    sweep,
        "Feature Set":              feature_set,
        "Threshold":                threshold,
        "Static Time (s)":          static_time,
        "Static Mem (MB)":          static_mem,
        "Static Val F1":            val_f1,
        "Static Val PR-AUC":        val_prauc,
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


def build_mlp_variation_specs(targets, mlp_variations, seed, var, mode):
    """Expand (target × MLP variation) into a flat list of sweep specs.

    Returns a list of (sweep_key, name, cfg) tuples — one per variation per
    target — so Phase 2.5 tunes the MLP head on every target (Champion AND
    Challenger), not just the last one.
    """
    specs = []
    for base_cfg, base_key, target_role in targets:
        clean_prefix = base_key.replace("Grid: ", "").split(" (Seed")[0]
        for var_name, var_settings in mlp_variations:
            cfg_tuned = Config(
                use_mlp_head=True,
                use_multiscale_prop=base_cfg.use_multiscale_prop,
                sgc_k=base_cfg.sgc_k,
                use_directional_prop=base_cfg.use_directional_prop,
                use_graph_structural=base_cfg.use_graph_structural,
                topo_injection_mode=base_cfg.topo_injection_mode,
                seed=base_cfg.seed,
                mlp_hidden=var_settings["mlp_hidden"],
                use_layernorm=False,
                use_residual=var_settings["use_residual"],
            )
            name = f"MLP-{var_name} [{clean_prefix}]"
            sweep_key = f"{name} (Seed {seed}, Var {var})" if mode == "mega" else name
            specs.append((sweep_key, name, cfg_tuned))
    return specs


def run_single_sweep(
    name: str,
    cfg: Config,
    df: pd.DataFrame,
    df_edge: pd.DataFrame,
    feature_cols: list,
    window: int = None,
    variation: str = "Base",
    only_static: bool = False,
    only_wf: bool = False,
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

    stat_res, static_f1, static_prauc = {}, 0.0, 0.0
    if not only_wf:
        with profile_resources() as stat_res:
            model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
            model.eval()
            with torch.no_grad():
                m      = (yte_g != -1)
                scores = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()

            y_true       = yte_g[m].numpy()
            static_f1    = f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)
            static_prauc = average_precision_score(y_true, scores)

    wf_res, wf_f1, wf_prauc, wf_records = {}, 0.0, 0.0, []
    if not only_static:
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
    if not only_wf:
        torch.save(model.state_dict(), os.path.join(model_dir, f"{safe_name}_model.pt"))
    if not only_static:
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
    feature_set: str = "N/A",
    threshold: str = "0.5",
) -> dict:
    """Run a fast static sweep with both val and OOT evaluation.
    
    P0-C: val_steps evaluation is used for model selection.
    test_steps evaluation is reported but NOT used for selection.
    """
    set_global_seeds(cfg.seed)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
    Xval_g, yval_g = stack_prop(dm, list(cfg.val_steps))
    Xte_g, yte_g = stack_prop(dm, list(cfg.test_steps))

    if variation == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=cfg.seed)
        Xtr_g = torch.tensor(pca.fit_transform(Xtr_g.numpy()), dtype=torch.float32)
        Xval_g = torch.tensor(pca.transform(Xval_g.numpy()), dtype=torch.float32)
        Xte_g = torch.tensor(pca.transform(Xte_g.numpy()), dtype=torch.float32)
    elif variation == "RF_Pruned":
        from sklearn.ensemble import RandomForestClassifier
        m_tr = (ytr_g != -1)
        rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=cfg.seed)
        rf.fit(Xtr_g[m_tr].numpy(), ytr_g[m_tr].numpy())
        mask = rf.feature_importances_ > 0.000
        if mask.sum() > 0:
            Xtr_g = Xtr_g[:, mask]
            Xval_g = Xval_g[:, mask]
            Xte_g = Xte_g[:, mask]

    valid_ytr = ytr_g[ytr_g != -1]
    counts    = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
    cls_w     = (counts.sum() / (2.0 * counts)).to(DEVICE)

    with profile_resources() as stat_res:
        model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
        model.eval()
        
        # Val evaluation (P0-C: used for model selection)
        with torch.no_grad():
            m_val = (yval_g != -1)
            if m_val.sum() > 0:
                s_val = torch.softmax(model(Xval_g[m_val].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                y_val_true = yval_g[m_val].numpy()
                val_f1 = f1_score(y_val_true, (s_val >= 0.5).astype(int), pos_label=1, zero_division=0)
                val_prauc = average_precision_score(y_val_true, s_val)
            else:
                val_f1, val_prauc = 0.0, 0.0

        # Test evaluation (reported but NOT used for selection)
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
        feature_set=feature_set,
        threshold=threshold,
        val_f1=round(val_f1, 3),
        val_prauc=round(val_prauc, 3),
    )


def walk_forward_baseline(
    dm: EllipticDataModule, cfg: Config, model_cls, sweep_name: str, 
    window: int = None, use_prop: bool = False, use_temporal: bool = False, 
    eval_steps: range = None, label_lag: int = 0, **model_kwargs
) -> tuple:
    """Walk-forward evaluation for scikit-learn/XGBoost tabular models."""
    from data.temporal_features import build_snapshot_temporal_features
    
    if eval_steps is None:
        eval_steps = cfg.test_steps
        
    y_true_all = []
    y_pred_all = []
    y_score_all = []
    wf_steps = []
    wf_f1_per_step = []
    wf_prauc_per_step = []
    
    for tau in eval_steps:
        # Train on [1, tau-1]
        Xs_tr, ys_tr = [], []
        t_start = max(1, tau - window) if window else 1
        for t in range(t_start, tau):
            g = dm.graphs[t]
            m = g["labeled_mask"].numpy()
            if m.sum() > 0:
                feat = g["prop"].numpy()[m] if use_prop else g["x"].numpy()[:, :166][m]
                if use_temporal:
                    # Append temporal lag features
                    t_window = 4 if window is None else window
                    t_feats = build_snapshot_temporal_features(dm, target_step=t, window=t_window, label_lag=label_lag)
                    t_feats_bcast = np.tile(t_feats, (feat.shape[0], 1))
                    feat = np.hstack([feat, t_feats_bcast])
                Xs_tr.append(feat)
                ys_tr.append(g["y"].numpy()[m])
                
        if len(Xs_tr) == 0:
            continue
            
        Xtr = np.concatenate(Xs_tr)
        ytr = np.concatenate(ys_tr)
        
        if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
            continue
            
        if "scale_pos_weight" in model_kwargs:
            model_kwargs["scale_pos_weight"] = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            
        # Test on tau
        g_tau = dm.graphs[tau]
        m_tau = g_tau["labeled_mask"].numpy()
        if m_tau.sum() == 0:
            continue
            
        Xte = g_tau["prop"].numpy()[m_tau] if use_prop else g_tau["x"].numpy()[:, :166][m_tau]
        if use_temporal:
            t_window = 4 if window is None else window
            t_feats_te = build_snapshot_temporal_features(dm, target_step=tau, window=t_window, label_lag=label_lag)
            t_feats_te_bcast = np.tile(t_feats_te, (Xte.shape[0], 1))
            Xte = np.hstack([Xte, t_feats_te_bcast])
            
        yte = g_tau["y"].numpy()[m_tau]
        
        # Train & Predict
        model = model_cls(**model_kwargs).fit(Xtr, ytr)
        s = model.predict_proba(Xte)[:, 1]
        y_pred = (s >= 0.5).astype(int)
        
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        y_score_all.append(s)
        
        from sklearn.metrics import f1_score, average_precision_score
        step_f1 = float(f1_score(yte, y_pred, pos_label=1, zero_division=0))
        step_prauc = float(average_precision_score(yte, s))
        
        wf_steps.append(tau)
        wf_f1_per_step.append(step_f1)
        wf_prauc_per_step.append(step_prauc)
            
    from evaluation.validation import _aggregate_walk_forward
    pooled_f1, pooled_prauc, macro_f1, macro_prauc, pooled_pak = _aggregate_walk_forward(y_true_all, y_pred_all, y_score_all)
    
    import pandas as pd
    import os
    from config import OUTPUT_DIR
    csv_file = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
    df_export = pd.DataFrame({
        "Sweep": [sweep_name] * len(wf_steps),
        "Timestep (tau)": wf_steps,
        "F1": wf_f1_per_step,
        "PR-AUC": wf_prauc_per_step
    })
    if os.path.exists(csv_file):
        df_export.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_export.to_csv(csv_file, index=False)
        
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
        # THRESHOLD FIX (P0-B): use training prevalence, not test labels
        ytr_labeled = np.concatenate([
            dm.graphs[t]["y"].numpy()[dm.graphs[t]["labeled_mask"].numpy()]
            for t in train_block
        ])
        train_rate = (ytr_labeled == 1).mean()
        y_pred = (scores >= np.percentile(scores, (1 - train_rate) * 100)).astype(int)
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
            
        # THRESHOLD FIX (P0-B): use training prevalence, not test labels
        ytr_labeled = np.concatenate([
            dm.graphs[t]["y"].numpy()[dm.graphs[t]["labeled_mask"].numpy()]
            for t in train_block
        ])
        train_rate = (ytr_labeled == 1).mean()
        thresh_pct = (1 - train_rate) * 100
        y_pred = (scores >= np.percentile(scores, thresh_pct)).astype(int)
        
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        y_score_all.append(scores)
        
    pooled_f1, pooled_prauc, _, _, _ = _aggregate_walk_forward(y_true_all, y_pred_all, y_score_all)
    return pooled_f1, pooled_prauc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "mega", "temporal"])
    parser.add_argument("--only-static", action="store_true", help="Run only static OOT")
    parser.add_argument("--only-wf", action="store_true", help="Run only walk-forward")
    args = parser.parse_args()

    from data.load_dataset import download_and_load_data

    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg_default = Config()

    results = []
    completed_sweeps = set()
    out_file = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    
    timestep_csv = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
    if os.path.exists(timestep_csv):
        os.remove(timestep_csv)
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
        spw_b = (ytr_b == 0).sum() / max((ytr_b == 1).sum(), 1)

        Xs_te, ys_te = [], []
        for t in cfg_tabular.test_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_te.append(g["x"].numpy()[:, :166][m])
            ys_te.append(g["y"].numpy()[m])
        Xte_b, yte_b = np.concatenate(Xs_te), np.concatenate(ys_te)

        if args.mode != "temporal" and "Baseline: IsolationForest (166)" not in completed_sweeps:
            from sklearn.ensemble import IsolationForest
            stat_iso, static_iso_f1, static_iso_prauc = {}, 0.0, 0.0
            if not args.only_wf:
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
                
            wf_iso, wf_iso_f1, wf_iso_prauc = {}, 0.0, 0.0
            if not args.only_static:
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
            if args.mode != "temporal": print("Already completed Baseline: IsolationForest (166), skipping.")



        if args.mode != "temporal" and "Baseline: XGBoost (166)" not in completed_sweeps:
            stat_xgb, static_xgb_f1, static_xgb_prauc = {}, 0.0, 0.0
            if not args.only_wf:
                with profile_resources() as stat_xgb:
                    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     scale_pos_weight=spw_b, eval_metric="aucpr",
                                     random_state=cfg_tabular.seed, n_jobs=1).fit(Xtr_b, ytr_b)
                s_xgb = xgb.predict_proba(Xte_b)[:, 1]
                static_xgb_f1 = f1_score(yte_b, (s_xgb >= 0.5).astype(int), pos_label=1)
                static_xgb_prauc = average_precision_score(yte_b, s_xgb)
                
            wf_xgb, wf_xgb_f1, wf_xgb_prauc = {}, 0.0, 0.0
            if not args.only_static:
                with profile_resources() as wf_xgb:
                    wf_xgb_f1, wf_xgb_prauc = walk_forward_baseline(
                        dm_base, cfg_tabular, XGBClassifier, sweep_name="Baseline: XGBoost (166)", window=None,
                    n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw_b, eval_metric="aucpr", random_state=cfg_tabular.seed, n_jobs=1
                )
            
            os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
            if not args.only_wf:
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
            if args.mode != "temporal": print("Already completed Baseline: XGBoost (166), skipping.")

        if args.mode != "temporal" and "Baseline: RandomForest (166)" not in completed_sweeps:
            stat_rf, static_rf_f1, static_rf_prauc = {}, 0.0, 0.0
            if not args.only_wf:
                with profile_resources() as stat_rf:
                    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            n_jobs=1, random_state=cfg_tabular.seed).fit(Xtr_b, ytr_b)
                s_rf = rf.predict_proba(Xte_b)[:, 1]
                static_rf_f1 = f1_score(yte_b, (s_rf >= 0.5).astype(int), pos_label=1)
                static_rf_prauc = average_precision_score(yte_b, s_rf)
                
            wf_rf, wf_rf_f1, wf_rf_prauc = {}, 0.0, 0.0
            if not args.only_static:
                with profile_resources() as wf_rf:
                    wf_rf_f1, wf_rf_prauc = walk_forward_baseline(
                        dm_base, cfg_tabular, RandomForestClassifier, sweep_name="Baseline: RandomForest (166)", window=None,
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
            if args.mode != "temporal": print("Already completed Baseline: RandomForest (166), skipping.")
            
        if args.mode != "temporal" and "Ablation: XGBoost + Graph Features (500d)" not in completed_sweeps:
            dm_graph = EllipticDataModule(df, df_edge, feature_cols, cfg_default)
            dm_graph.setup()
            
            Xs_tr_g, ys_tr_g = [], []
            for t in cfg_default.train_steps:
                g = dm_graph.graphs[t]; m = g["labeled_mask"].numpy()
                Xs_tr_g.append(g["prop"].numpy()[m])
                ys_tr_g.append(g["y"].numpy()[m])
            Xtr_g, ytr_g = np.concatenate(Xs_tr_g), np.concatenate(ys_tr_g)
            
            Xs_te_g, ys_te_g = [], []
            for t in cfg_default.test_steps:
                g = dm_graph.graphs[t]; m = g["labeled_mask"].numpy()
                Xs_te_g.append(g["prop"].numpy()[m])
                ys_te_g.append(g["y"].numpy()[m])
            Xte_g, yte_g = np.concatenate(Xs_te_g), np.concatenate(ys_te_g)

            spw_g = (ytr_g == 0).sum() / max((ytr_g == 1).sum(), 1)
            stat_xgb_g, static_xgb_g_f1, static_xgb_g_prauc = {}, 0.0, 0.0
            if not args.only_wf:
                with profile_resources() as stat_xgb_g:
                    xgb_g = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     scale_pos_weight=spw_g, eval_metric="aucpr", tree_method="hist",
                                     random_state=cfg_default.seed, n_jobs=1).fit(Xtr_g, ytr_g)
                s_xgb_g = xgb_g.predict_proba(Xte_g)[:, 1]
                static_xgb_g_f1 = f1_score(yte_g, (s_xgb_g >= 0.5).astype(int), pos_label=1)
                static_xgb_g_prauc = average_precision_score(yte_g, s_xgb_g)
                
            wf_xgb_g, wf_xgb_g_f1, wf_xgb_g_prauc = {}, 0.0, 0.0
            if not args.only_static:
                with profile_resources() as wf_xgb_g:
                    wf_xgb_g_f1, wf_xgb_g_prauc = walk_forward_baseline(
                        dm_graph, cfg_default, XGBClassifier, window=None, use_prop=True,
                    n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw_g, eval_metric="aucpr", tree_method="hist", random_state=cfg_default.seed, n_jobs=1
                )
            
            results.append(_make_result(
                seed=cfg_default.seed,
                variation="Base",
                sweep="Ablation: XGBoost + Graph Features (500d)",
                static_time=round(stat_xgb_g.get("time", 0.0), 3),
                static_mem=round(stat_xgb_g.get("peak_mem", 0.0), 2),
                static_f1=round(static_xgb_g_f1, 3),
                static_prauc=round(static_xgb_g_prauc, 3),
                wf_time=round(wf_xgb_g.get("time", 0.0), 3),
                wf_mem=round(wf_xgb_g.get("peak_mem", 0.0), 2),
                wf_f1=round(wf_xgb_g_f1, 3),
                wf_prauc=round(wf_xgb_g_prauc, 3),
            ))
            pd.DataFrame(results).to_csv(out_file, index=False)
        else:
            if args.mode != "temporal": print("Already completed Ablation: XGBoost + Graph Features (500d), skipping.")
            
        if "Baseline: Temporal XGBoost (best w, lag=0)" not in completed_sweeps:
            if not args.only_static:
                # 1. Tune w on val_steps
                best_w = 4
                best_val_f1 = -1.0
                for w in [1, 2, 4]:
                    val_f1, _ = walk_forward_baseline(
                        dm_base, cfg_tabular, XGBClassifier, sweep_name=f"TempXGB-val-w{w}",
                        window=None, use_prop=False, use_temporal=True, eval_steps=cfg_tabular.val_steps, label_lag=0,
                        n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw_b, 
                        eval_metric="aucpr", tree_method="hist", random_state=cfg_tabular.seed, n_jobs=1
                    )
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_w = w
                
                # 2. Evaluate best_w on test_steps with lag=0
                with profile_resources() as wf_temp_0:
                    wf_temp_xgb_f1_0, wf_temp_xgb_prauc_0 = walk_forward_baseline(
                        dm_base, cfg_tabular, XGBClassifier, sweep_name="Baseline: Temporal XGBoost (lag=0)",
                        window=None, use_prop=False, use_temporal=True, eval_steps=cfg_tabular.test_steps, label_lag=0,
                        n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw_b, 
                        eval_metric="aucpr", tree_method="hist", random_state=cfg_tabular.seed, n_jobs=1
                    )
                    
                # 3. Evaluate best_w on test_steps with lag=2
                with profile_resources() as wf_temp_2:
                    wf_temp_xgb_f1_2, wf_temp_xgb_prauc_2 = walk_forward_baseline(
                        dm_base, cfg_tabular, XGBClassifier, sweep_name="Baseline: Temporal XGBoost (lag=2)",
                        window=None, use_prop=False, use_temporal=True, eval_steps=cfg_tabular.test_steps, label_lag=2,
                        n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw_b, 
                        eval_metric="aucpr", tree_method="hist", random_state=cfg_tabular.seed, n_jobs=1
                    )
            
                results.append(_make_result(
                    seed=cfg_tabular.seed, variation="Base", sweep="Baseline: Temporal XGBoost (best w, lag=0)",
                    static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
                    wf_time=round(wf_temp_0.get("time", 0.0), 3), wf_mem=round(wf_temp_0.get("peak_mem", 0.0), 2),
                    wf_f1=round(wf_temp_xgb_f1_0, 3), wf_prauc=round(wf_temp_xgb_prauc_0, 3),
                    feature_set=f"Raw-166+Temporal (w={best_w})", threshold="0.5"
                ))
                results.append(_make_result(
                    seed=cfg_tabular.seed, variation="Base", sweep="Baseline: Temporal XGBoost (best w, lag=2)",
                    static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
                    wf_time=round(wf_temp_2.get("time", 0.0), 3), wf_mem=round(wf_temp_2.get("peak_mem", 0.0), 2),
                    wf_f1=round(wf_temp_xgb_f1_2, 3), wf_prauc=round(wf_temp_xgb_prauc_2, 3),
                    feature_set=f"Raw-166+Temporal-Lag2 (w={best_w})", threshold="0.5"
                ))
                pd.DataFrame(results).to_csv(out_file, index=False)
        else:
            print("Already completed Baseline: Temporal XGBoost, skipping.")
            
    except Exception as e:
        print(f"  Baselines skipped: {e}")

    # ── W4 FIX: Phased Grid Search Matrix ──────────────────────────────────────
    import itertools
    if args.mode == "temporal":
        seeds = []
        variations = []
        k_vals = []
        dir_vals = []
        topo_vals = []
    elif args.mode == "mega":
        seeds = [42, 43, 44]
        variations = ["Base", "PCA", "RF_Pruned"]
        k_vals = [1, 2, 3]
        dir_vals = [False, True]
        topo_vals = [None, 'late', 'early']
    else:
        seeds = [42]
        variations = ["Base"]
        k_vals = [1, 2, 3]
        dir_vals = [False, True]
        topo_vals = [None, 'late', 'early']

    # Helper to execute and record a single sweep configuration
    def execute_sweep(sweep_key, name, cfg, var):
        if sweep_key in completed_sweeps:
            print(f"Already completed {sweep_key}, skipping.")
            for r in results:
                if r["Sweep"] == sweep_key: return r
            return None
            
        print(f"\n{'='*55}\nRunning: {sweep_key}\n{'='*55}")
        
        if args.only_static:
            res = run_static_only_sweep(name, cfg, df, df_edge, feature_cols, variation=var)
        elif args.only_wf:
            res = run_single_sweep(name, cfg, df, df_edge, feature_cols, variation=var, only_wf=True)
        else:
            # During Phase 1 and Phase 2, we ONLY want to evaluate statically, regardless of mode.
            res = run_static_only_sweep(name, cfg, df, df_edge, feature_cols, variation=var)
                
        res["Sweep"] = sweep_key
        results.append(res)
        pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
        print(f"--> {res}\n")
        completed_sweeps.add(sweep_key)
        return res

    all_configs_run = {}

    for seed in seeds:
        for var in variations:
            print(f"\n{'#'*60}\nRunning Sequence for Seed {seed}, Var {var}\n{'#'*60}")
            
            # --- PHASE 1: Baselines ---
            phase1_sweeps = [
                ("Sweep 1: SGC (baseline)", Config(use_mlp_head=False, use_multiscale_prop=False, use_graph_structural=False, seed=seed)),
                ("Sweep 2: + MLP Head", Config(use_mlp_head=True, use_multiscale_prop=False, use_graph_structural=False, seed=seed))
            ]
            for name, cfg in phase1_sweeps:
                sweep_key = f"{name} (Seed {seed}, Var {var})" if args.mode == "mega" else name
                execute_sweep(sweep_key, name, cfg, var)
                all_configs_run[sweep_key] = cfg
                
            # --- PHASE 2: Grid Search ---
            best_grid_key = None
            best_grid_f1 = -1.0
            
            for k, directional, topo, injection in itertools.product([1, 2, 3], [False, True], [False, True], ['late', 'early']):
                if not topo and injection == 'early':
                    continue # Avoid duplicates: if no topo, injection mode doesn't matter
                    
                cfg = Config(
                    use_mlp_head=True,
                    use_multiscale_prop=True,
                    sgc_k=k,
                    use_directional_prop=directional,
                    use_graph_structural=topo,
                    topo_injection_mode=injection,
                    seed=seed
                )
                
                name_parts = [f"K={k}"]
                name_parts.append("Dir=T" if directional else "Dir=F")
                if topo:
                    name_parts.append(f"Topo={injection}")
                else:
                    name_parts.append("Topo=None")
                    
                name = f"Grid: {', '.join(name_parts)}"
                sweep_key = f"{name} (Seed {seed}, Var {var})" if args.mode == "mega" else name
                
                res = execute_sweep(sweep_key, name, cfg, var)
                all_configs_run[sweep_key] = cfg
                if res and not args.only_wf:
                    # P0-C: Select on val F1, NOT test OOT F1
                    f1_val = res.get("Static Val F1", 0.0)
                    if pd.notna(f1_val) and isinstance(f1_val, (int, float)) and f1_val > best_grid_f1:
                        best_grid_f1 = f1_val
                        best_grid_key = sweep_key
                        
            print(f"\n  [Phase 2 Winner] {best_grid_key} achieved highest Val F1: {best_grid_f1:.3f}.")
            
            # --- PHASE 2.5: MLP Head Tuning (Champion & Challenger) ---
            print("\n--- PHASE 2.5: MLP Head Tuning ---")
            champion_key = best_grid_key
            champion_cfg = all_configs_run.get(champion_key)
            if champion_cfg is None:
                champion_cfg = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=1, use_directional_prop=False, use_graph_structural=False, seed=seed)
                champion_key = "Grid: K=1, Dir=F, Topo=None"

            # Challenger is highest dimensional config: K=3, Dir=T, Topo=early
            challenger_key = f"Grid: K=3, Dir=T, Topo=early (Seed {seed}, Var {var})" if args.mode == "mega" else "Grid: K=3, Dir=T, Topo=early"
            challenger_cfg = all_configs_run.get(challenger_key)
            if challenger_cfg is None:
                challenger_cfg = Config(
                    use_mlp_head=True,
                    use_multiscale_prop=True,
                    sgc_k=3,
                    use_directional_prop=True,
                    use_graph_structural=True,
                    topo_injection_mode='early',
                    seed=seed
                )

            if args.mode != "temporal":
                targets = [(champion_cfg, champion_key, "Champion")]
                if challenger_key != champion_key:
                    targets.append((challenger_cfg, challenger_key, "Challenger"))

                mlp_variations = [
                    ("Wide", {"mlp_hidden": (512, 256, 128), "use_residual": False}),
                    ("Residual", {"mlp_hidden": (128, 64), "use_residual": True}),
                    ("ResWide", {"mlp_hidden": (512, 256, 128), "use_residual": True})
                ]

                for _, base_key, target_role in targets:
                    clean_prefix = base_key.replace("Grid: ", "").split(" (Seed")[0]
                    print(f"Running MLP variations for {target_role} ({clean_prefix})...")

                specs = build_mlp_variation_specs(targets, mlp_variations, seed, var, args.mode)
                for sweep_key, name, cfg_tuned in specs:
                    execute_sweep(sweep_key, name, cfg_tuned, var)
                    all_configs_run[sweep_key] = cfg_tuned
            

    # ── PHASE 3: Walk-Forward on Best SGC Configuration ──────────────────────
    if args.mode != "temporal":
        print("\n--- Walk-Forward Validation (Best SGC Sweep) ---")
    best_f1 = -1.0
    best_sweep_name = None
    for r in results:
        # Check if it's an SGC sweep and has valid F1 (including MLP variants)
        if isinstance(r.get("Sweep"), str) and ("Sweep " in r["Sweep"] or "Grid: " in r["Sweep"] or "MLP-" in r["Sweep"]):
            f1_val = r.get("Static Val F1", "")
            if f1_val == "" or pd.isna(f1_val):
                f1_val = r.get("Static OOT F1", 0.0)
            if f1_val == "" or pd.isna(f1_val):
                f1_val = 0.0
            f1_val = float(f1_val)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_sweep_name = r["Sweep"]

    if best_sweep_name:
        wf_name = f"Best WF: {best_sweep_name}"
        if args.mode != "temporal" and wf_name not in completed_sweeps:
            print(f"\nWinning Configuration: {best_sweep_name} (Val F1: {best_f1:.3f})")
            best_cfg = all_configs_run.get(best_sweep_name)
            
            # If we skipped the execute loop (e.g. in temporal mode), best_cfg might be None.
            # We can load it from disk where run_static_only_sweep saves it.
            if best_cfg is None:
                safe_name = re.sub(r"[^\w\-]", "_", best_sweep_name)
                cfg_path = os.path.join(OUTPUT_DIR, "models", f"{safe_name}_cfg.pkl")
                if not os.path.exists(cfg_path):
                    raise FileNotFoundError(
                        f"Cannot run Phase 3 walk-forward: config for best sweep "
                        f"'{best_sweep_name}' was neither retained in memory nor persisted at "
                        f"{cfg_path}. Ensure it ran through run_static_only_sweep this session."
                    )
                best_cfg = joblib.load(cfg_path)
            
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
                    feature_set="Prop-N",
                    threshold="τ-1 calibrated",
                )
                results.append(wf_res)
                pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
                print(f"--> {wf_res}\n")
        else:
            if args.mode != "temporal": print(f"Already completed {wf_name}, skipping.")
            
        # Get best_cfg for EMA/LSTM if not already loaded
        best_cfg = all_configs_run.get(best_sweep_name)
        if best_cfg is None:
            import joblib
            safe_name = re.sub(r"[^\w\-]", "_", best_sweep_name)
            best_cfg = joblib.load(os.path.join(OUTPUT_DIR, "models", f"{safe_name}_cfg.pkl"))
            
        # Add EMA and LSTM
        ema_name = "EMA-Conditioned SGC Head"
        if ema_name not in completed_sweeps:
            print(f"\nRunning {ema_name} based on best SGC config...")
            with profile_resources() as ema_stat:
                dm_best = EllipticDataModule(df, df_edge, feature_cols, best_cfg)
                dm_best.setup()
                ema_f1, ema_prauc = walk_forward_ema_conditioned(dm_best, best_cfg, DEVICE, sweep_name=ema_name)
            
            ema_res = _make_result(
                seed=best_cfg.seed, variation="Base", sweep=ema_name,
                static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
                wf_time=round(ema_stat.get("time", 0.0), 3), wf_mem=round(ema_stat.get("peak_mem", 0.0), 2),
                wf_f1=round(ema_f1, 3), wf_prauc=round(ema_prauc, 3),
                feature_set="Prop+EMA-h", threshold="τ-1 calibrated"
            )
            results.append(ema_res)
            pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
            
        lstm_name = "LSTM-Conditioned SGC Head"
        if lstm_name not in completed_sweeps:
            print(f"\nRunning {lstm_name} based on best SGC config...")
            with profile_resources() as lstm_stat:
                dm_best = EllipticDataModule(df, df_edge, feature_cols, best_cfg)
                dm_best.setup()
                lstm_device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
                lstm_f1, lstm_prauc = walk_forward_lstm_conditioned(dm_best, best_cfg, lstm_device, sweep_name=lstm_name)
                
            lstm_res = _make_result(
                seed=best_cfg.seed, variation="Base", sweep=lstm_name,
                static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
                wf_time=round(lstm_stat.get("time", 0.0), 3), wf_mem=round(lstm_stat.get("peak_mem", 0.0), 2),
                wf_f1=round(lstm_f1, 3), wf_prauc=round(lstm_prauc, 3),
                feature_set="Prop+LSTM-h", threshold="τ-1 calibrated"
            )
            results.append(lstm_res)
            pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)


    df_res = pd.DataFrame(results, columns=list(_RESULT_KEYS))
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
