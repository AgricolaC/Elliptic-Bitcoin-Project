"""
run_sweeps.py — Ablation sweep runner.

W4 FIX: Expanded ablation matrix that isolates each mechanism independently.
W5 FIX: All result dicts use the standardized key schema (see _RESULT_KEYS).
        v2 (CSV-1): regime-stratified WF columns (Pre43/Shock/Recovery, pooled +
        macro), Threshold_Method, and Selfcond_Bug provenance.
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
    "Feature_Set",
    "Threshold_Method",
    "Static_Time_s",
    "Static_Mem_MB",
    "Static_Val_Pooled_F1",
    "Static_Val_Pooled_PRAUC",
    "Static_Val_Macro_F1",
    "Static_Val_Macro_PRAUC",
    "Static_OOT_Pooled_F1",
    "Static_OOT_Pooled_PRAUC",
    "Static_OOT_Macro_F1",
    "Static_OOT_Macro_PRAUC",
    "WF_Time_s",
    "WF_Mem_MB",
    "WF_Pooled_F1",
    "WF_Pooled_PRAUC",
    "WF_Macro_F1",
    "WF_Macro_PRAUC",
    "WF_Pre43_Pooled_F1",
    "WF_Pre43_PRAUC",
    "WF_Shock_F1",
    "WF_Shock_PRAUC",
    "WF_Recovery_Pooled_F1",
    "WF_Recovery_PRAUC",
    "Selfcond_Bug",
    "Notes",
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



def _evaluate_static_macro(predict_fn, dm, steps):
    from evaluation.validation import _aggregate_walk_forward
    y_true_list, y_pred_list, score_list = [], [], []
    for t in steps:
        if t not in dm.graphs: continue
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        if m.sum() == 0: continue
        y = g["y"].numpy()[m]
        if len(np.unique(y)) < 2: continue
        s = predict_fn(g, m)
        y_pred = (s >= 0.5).astype(int)
        y_true_list.append(y)
        y_pred_list.append(y_pred)
        score_list.append(s)
    if not y_true_list: return "N/A", "N/A"
    _, _, macro_f1, macro_prauc, _ = _aggregate_walk_forward(y_true_list, y_pred_list, score_list)
    return float(macro_f1), float(macro_prauc)

def _make_result(
    seed: int,
    variation: str,
    sweep: str,
    static_time: float | str,
    static_mem: float | str,
    static_oot_pooled_f1: float | str,
    static_oot_pooled_prauc: float | str,
    wf_time: float | str,
    wf_mem: float | str,
    wf_f1: float | str,            # back-compat: old "Walk-Forward Mean" → WF_Macro
    wf_prauc: float | str,
    feature_set: str = "N/A",
    threshold: str = "0.5",        # back-compat alias for threshold_method
    static_val_pooled_f1: float | str = "N/A",
    static_val_pooled_prauc: float | str = "N/A",
    # ── v2 schema additions (CSV-1) ──────────────────────────────────────────
    threshold_method: str | None = None,
    static_val_macro_f1: float | str = "N/A",
    static_val_macro_prauc: float | str = "N/A",
    static_oot_macro_f1: float | str = "N/A",
    static_oot_macro_prauc: float | str = "N/A",
    wf_pooled_f1: float | str = "N/A",
    wf_pooled_prauc: float | str = "N/A",
    wf_pre43_pooled_f1: float | str = "N/A",
    wf_pre43_prauc: float | str = "N/A",
    wf_shock_f1: float | str = "N/A",
    wf_shock_prauc: float | str = "N/A",
    wf_recovery_pooled_f1: float | str = "N/A",
    wf_recovery_prauc: float | str = "N/A",
    selfcond_bug: str = "fixed",
    notes: str = "",
) -> dict:
    """
    Construct a result dict with the standardized v2 key schema (CSV-1).

    Back-compat: the legacy ``wf_f1``/``wf_prauc`` params (the old single
    "Walk-Forward Mean") map to the MACRO columns; ``threshold`` aliases
    ``threshold_method``. New regime-stratified columns default to "N/A".
    ``selfcond_bug`` records provenance: "present" (pre-fix) | "fixed".
    Raises AssertionError if any key would be missing.
    """
    result = {
        "Seed":                  seed,
        "Variation":             variation,
        "Sweep":                 sweep,
        "Feature_Set":           feature_set,
        "Threshold_Method":      threshold_method if threshold_method is not None else threshold,
        "Static_Time_s":         static_time,
        "Static_Mem_MB":         static_mem,
        "Static_Val_Pooled_F1":  static_val_pooled_f1,
        "Static_Val_Pooled_PRAUC": static_val_pooled_prauc,
        "Static_Val_Macro_F1":   static_val_macro_f1,
        "Static_Val_Macro_PRAUC": static_val_macro_prauc,
        "Static_OOT_Pooled_F1":  static_oot_pooled_f1,
        "Static_OOT_Pooled_PRAUC": static_oot_pooled_prauc,
        "Static_OOT_Macro_F1":   static_oot_macro_f1,
        "Static_OOT_Macro_PRAUC": static_oot_macro_prauc,
        "WF_Time_s":             wf_time,
        "WF_Mem_MB":             wf_mem,
        "WF_Pooled_F1":          wf_pooled_f1,
        "WF_Pooled_PRAUC":       wf_pooled_prauc,
        "WF_Macro_F1":           wf_f1,
        "WF_Macro_PRAUC":        wf_prauc,
        "WF_Pre43_Pooled_F1":    wf_pre43_pooled_f1,
        "WF_Pre43_PRAUC":        wf_pre43_prauc,
        "WF_Shock_F1":           wf_shock_f1,
        "WF_Shock_PRAUC":        wf_shock_prauc,
        "WF_Recovery_Pooled_F1": wf_recovery_pooled_f1,
        "WF_Recovery_PRAUC":     wf_recovery_prauc,
        "Selfcond_Bug":          selfcond_bug,
        "Notes":                 notes,
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
    static_macro_f1, static_macro_prauc = "N/A", "N/A"
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
            
            def predict_fn(g_t, m_t):
                with torch.no_grad():
                    return torch.softmax(model(g_t["prop"][m_t].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            static_macro_f1, static_macro_prauc = _evaluate_static_macro(predict_fn, dm, cfg.test_steps)

    wf_res, wf_f1, wf_prauc, wf_records = {}, 0.0, 0.0, []
    if not only_static:
        with profile_resources() as wf_res:
            wf_f1, wf_prauc, wf_records = walk_forward_validation(
                dm, cfg, DEVICE, sweep_name=name, return_records=True, window=window,
            )

    safe_name = re.sub(r"[^\w\-]", "_", name)
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Dump static OOT model, and Walk-Forward records
    # joblib.dump(dm, os.path.join(model_dir, f"{safe_name}_dm.pkl"))
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
        static_oot_pooled_f1=round(static_f1, 3),
        static_oot_pooled_prauc=round(static_prauc, 3),
        static_oot_macro_f1=round(static_macro_f1, 3) if isinstance(static_macro_f1, float) else static_macro_f1,
        static_oot_macro_prauc=round(static_macro_prauc, 3) if isinstance(static_macro_prauc, float) else static_macro_prauc,
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



    valid_ytr = ytr_g[ytr_g != -1]
    counts    = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
    cls_w     = (counts.sum() / (2.0 * counts)).to(DEVICE)

    with profile_resources() as stat_res:
        model = fit_head(Xtr_g, ytr_g, Xtr_g.shape[1], cfg, cls_w, DEVICE)
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
        
        def predict_fn(g_t, m_t):
            with torch.no_grad():
                return torch.softmax(model(g_t["prop"][m_t].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        static_macro_f1, static_macro_prauc = _evaluate_static_macro(predict_fn, dm, cfg.test_steps)
        
        if len(cfg.val_steps) > 0:
            val_macro_f1, val_macro_prauc = _evaluate_static_macro(predict_fn, dm, cfg.val_steps)
        else:
            val_macro_f1, val_macro_prauc = "N/A", "N/A"
    
    # Save the static-only model + dm + cfg for potential later analysis
    safe_name = re.sub(r"[^\w\-]", "_", name)
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    # joblib.dump(dm, os.path.join(model_dir, f"{safe_name}_dm.pkl"))
    joblib.dump(cfg, os.path.join(model_dir, f"{safe_name}_cfg.pkl"))
    torch.save(model.state_dict(), os.path.join(model_dir, f"{safe_name}_model.pt"))

    return _make_result(
        seed=cfg.seed,
        variation=variation,
        sweep=name,
        static_time=round(stat_res.get("time", 0.0), 3),
        static_mem=round(stat_res.get("peak_mem", 0.0), 2),
        static_oot_pooled_f1=round(static_f1, 3),
        static_oot_pooled_prauc=round(static_prauc, 3),
        static_oot_macro_f1=round(static_macro_f1, 3) if isinstance(static_macro_f1, float) else static_macro_f1,
        static_oot_macro_prauc=round(static_macro_prauc, 3) if isinstance(static_macro_prauc, float) else static_macro_prauc,
        wf_time="N/A",
        wf_mem="N/A",
        wf_f1="N/A",
        wf_prauc="N/A",
        feature_set=feature_set,
        threshold=threshold,
        static_val_pooled_f1=round(val_f1, 3),
        static_val_pooled_prauc=round(val_prauc, 3),
        static_val_macro_f1=round(val_macro_f1, 3) if isinstance(val_macro_f1, float) else val_macro_f1,
        static_val_macro_prauc=round(val_macro_prauc, 3) if isinstance(val_macro_prauc, float) else val_macro_prauc,
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
            
    from evaluation.wf_metrics import stratified_wf_metrics
    from evaluation.ablation_validation import _write_csv2
    records_dict = [{"tau": t, "y_true": yt, "scores": ys, "y_pred": yp} 
                    for t, yt, ys, yp in zip(wf_steps, y_true_all, y_score_all, y_pred_all)]
    agg, rows = stratified_wf_metrics(records_dict, threshold=0.5)
    _write_csv2(sweep_name, rows, {})
    return agg


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
    from evaluation.wf_metrics import stratified_wf_metrics
    from evaluation.ablation_validation import _write_csv2
    records_dict = [{"tau": t, "y_true": yt, "scores": ys, "y_pred": yp} 
                    for t, yt, ys, yp in zip(cfg.test_steps[-len(y_true_all):], y_true_all, y_score_all, y_pred_all)]
    agg, rows = stratified_wf_metrics(records_dict, threshold=0.5)
    _write_csv2("Baseline: IsolationForest (166)", rows, {})
    return agg

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
        
    from evaluation.wf_metrics import stratified_wf_metrics
    from evaluation.ablation_validation import _write_csv2
    records_dict = [{"tau": t, "y_true": yt, "scores": ys, "y_pred": yp} 
                    for t, yt, ys, yp in zip(cfg.test_steps[-len(y_true_all):], y_true_all, y_score_all, y_pred_all)]
    agg, rows = stratified_wf_metrics(records_dict, threshold=0.5)
    _write_csv2("Baseline: AutoEncoder (166)", rows, {})
    return agg




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "mega", "mega_k5", "temporal"])
    parser.add_argument("--only-static", action="store_true", help="Run only static OOT")
    parser.add_argument("--only-wf", action="store_true", help="Run only walk-forward")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    from data.load_dataset import download_and_load_data

    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg_default = Config(seed=args.seed)

    results = []
    completed_sweeps = set()
    out_file = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    
    timestep_csv = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
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
                iso_thresh = np.percentile(scores, thresh_pct)
                static_iso_f1 = f1_score(yte_b, (scores >= iso_thresh).astype(int), pos_label=1, zero_division=0)
                static_iso_prauc = average_precision_score(yte_b, scores)
                
                def predict_iso(g_t, m_t):
                    return -iso.score_samples(g_t["x"][m_t, :166].numpy())
                def _iso_thresh(g_t, m_t):
                    return np.percentile(-iso.score_samples(g_t["x"][m_t, :166].numpy()), thresh_pct)
                # ISO macro needs custom threshold logic, we will skip macro for ISO forest since it's unsupervised
                iso_oot_macro_f1, iso_oot_macro_prauc = "N/A", "N/A"
                iso_val_pooled_f1, iso_val_pooled_prauc = "N/A", "N/A"
                iso_val_macro_f1, iso_val_macro_prauc = "N/A", "N/A"
                
            results.append(_make_result(
                seed=cfg_tabular.seed,
                variation="Base",
                sweep="Baseline: IsolationForest (166)",
                static_time=round(stat_iso.get("time", 0.0), 3),
                static_mem=round(stat_iso.get("peak_mem", 0.0), 2),
                static_val_pooled_f1=iso_val_pooled_f1,
                static_val_pooled_prauc=iso_val_pooled_prauc,
                static_val_macro_f1=iso_val_macro_f1,
                static_val_macro_prauc=iso_val_macro_prauc,
                static_oot_pooled_f1=round(static_iso_f1, 3),
                static_oot_pooled_prauc=round(static_iso_prauc, 3),
                static_oot_macro_f1=iso_oot_macro_f1,
                static_oot_macro_prauc=iso_oot_macro_prauc,
                wf_time="N/A",
                wf_mem="N/A",
                wf_f1="N/A",
                wf_prauc="N/A",
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
                
                def predict_xgb(g_t, m_t):
                    return xgb.predict_proba(g_t["x"][m_t, :166].numpy())[:, 1]
                xgb_oot_macro_f1, xgb_oot_macro_prauc = _evaluate_static_macro(predict_xgb, dm_base, cfg_tabular.test_steps)
                if len(cfg_tabular.val_steps) > 0:
                    xgb_val_macro_f1, xgb_val_macro_prauc = _evaluate_static_macro(predict_xgb, dm_base, cfg_tabular.val_steps)
                    Xval_b = np.concatenate([dm_base.graphs[t]["x"].numpy()[:, :166] for t in cfg_tabular.val_steps])
                    yval_b = np.concatenate([dm_base.graphs[t]["y"].numpy() for t in cfg_tabular.val_steps])
                    m_val_b = yval_b != -1
                    if m_val_b.sum() > 0:
                        s_val_xgb = xgb.predict_proba(Xval_b[m_val_b])[:, 1]
                        xgb_val_pooled_f1 = f1_score(yval_b[m_val_b], (s_val_xgb >= 0.5).astype(int), pos_label=1)
                        xgb_val_pooled_prauc = average_precision_score(yval_b[m_val_b], s_val_xgb)
                    else:
                        xgb_val_pooled_f1, xgb_val_pooled_prauc = 0.0, 0.0
                else:
                    xgb_val_macro_f1, xgb_val_macro_prauc = "N/A", "N/A"
                    xgb_val_pooled_f1, xgb_val_pooled_prauc = "N/A", "N/A"
                
                wf_xgb, wf_xgb_agg = {}, None
            if not args.only_static:
                with profile_resources() as wf_xgb:
                    wf_xgb_agg = walk_forward_baseline(
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
                static_val_pooled_f1=round(xgb_val_pooled_f1, 3) if isinstance(xgb_val_pooled_f1, float) else xgb_val_pooled_f1,
                static_val_pooled_prauc=round(xgb_val_pooled_prauc, 3) if isinstance(xgb_val_pooled_prauc, float) else xgb_val_pooled_prauc,
                static_val_macro_f1=round(xgb_val_macro_f1, 3) if isinstance(xgb_val_macro_f1, float) else xgb_val_macro_f1,
                static_val_macro_prauc=round(xgb_val_macro_prauc, 3) if isinstance(xgb_val_macro_prauc, float) else xgb_val_macro_prauc,
                static_oot_pooled_f1=round(static_xgb_f1, 3),
                static_oot_pooled_prauc=round(static_xgb_prauc, 3),
                static_oot_macro_f1=round(xgb_oot_macro_f1, 3) if isinstance(xgb_oot_macro_f1, float) else xgb_oot_macro_f1,
                static_oot_macro_prauc=round(xgb_oot_macro_prauc, 3) if isinstance(xgb_oot_macro_prauc, float) else xgb_oot_macro_prauc,
                wf_time=round(wf_xgb.get("time", 0.0), 3),
                wf_mem=round(wf_xgb.get("peak_mem", 0.0), 2),
                wf_f1=round(wf_xgb_agg["WF_Macro_F1"], 3) if wf_xgb_agg else "N/A",
                wf_prauc=round(wf_xgb_agg["WF_Macro_PRAUC"], 3) if wf_xgb_agg else "N/A",
                wf_pooled_f1=round(wf_xgb_agg["WF_Pooled_F1"], 3) if wf_xgb_agg else "N/A",
                wf_pooled_prauc=round(wf_xgb_agg["WF_Pooled_PRAUC"], 3) if wf_xgb_agg else "N/A",
                wf_pre43_pooled_f1=round(wf_xgb_agg["WF_Pre43_Pooled_F1"], 3) if wf_xgb_agg else "N/A",
                wf_pre43_prauc=round(wf_xgb_agg["WF_Pre43_PRAUC"], 3) if wf_xgb_agg else "N/A",
                wf_shock_f1=round(wf_xgb_agg["WF_Shock_F1"], 3) if wf_xgb_agg else "N/A",
                wf_shock_prauc=round(wf_xgb_agg["WF_Shock_PRAUC"], 3) if wf_xgb_agg else "N/A",
                wf_recovery_pooled_f1=round(wf_xgb_agg["WF_Recovery_Pooled_F1"], 3) if wf_xgb_agg else "N/A",
                wf_recovery_prauc=round(wf_xgb_agg["WF_Recovery_PRAUC"], 3) if wf_xgb_agg else "N/A",
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
                
                def predict_rf(g_t, m_t):
                    return rf.predict_proba(g_t["x"][m_t, :166].numpy())[:, 1]
                rf_oot_macro_f1, rf_oot_macro_prauc = _evaluate_static_macro(predict_rf, dm_base, cfg_tabular.test_steps)
                if len(cfg_tabular.val_steps) > 0:
                    rf_val_macro_f1, rf_val_macro_prauc = _evaluate_static_macro(predict_rf, dm_base, cfg_tabular.val_steps)
                    if m_val_b.sum() > 0:
                        s_val_rf = rf.predict_proba(Xval_b[m_val_b])[:, 1]
                        rf_val_pooled_f1 = f1_score(yval_b[m_val_b], (s_val_rf >= 0.5).astype(int), pos_label=1)
                        rf_val_pooled_prauc = average_precision_score(yval_b[m_val_b], s_val_rf)
                    else:
                        rf_val_pooled_f1, rf_val_pooled_prauc = 0.0, 0.0
                else:
                    rf_val_macro_f1, rf_val_macro_prauc = "N/A", "N/A"
                    rf_val_pooled_f1, rf_val_pooled_prauc = "N/A", "N/A"
                
            results.append(_make_result(
                seed=cfg_tabular.seed,
                variation="Base",
                sweep="Baseline: RandomForest (166)",
                static_time=round(stat_rf.get("time", 0.0), 3),
                static_mem=round(stat_rf.get("peak_mem", 0.0), 2),
                static_val_pooled_f1=round(rf_val_pooled_f1, 3) if isinstance(rf_val_pooled_f1, float) else rf_val_pooled_f1,
                static_val_pooled_prauc=round(rf_val_pooled_prauc, 3) if isinstance(rf_val_pooled_prauc, float) else rf_val_pooled_prauc,
                static_val_macro_f1=round(rf_val_macro_f1, 3) if isinstance(rf_val_macro_f1, float) else rf_val_macro_f1,
                static_val_macro_prauc=round(rf_val_macro_prauc, 3) if isinstance(rf_val_macro_prauc, float) else rf_val_macro_prauc,
                static_oot_pooled_f1=round(static_rf_f1, 3),
                static_oot_pooled_prauc=round(static_rf_prauc, 3),
                static_oot_macro_f1=round(rf_oot_macro_f1, 3) if isinstance(rf_oot_macro_f1, float) else rf_oot_macro_f1,
                static_oot_macro_prauc=round(rf_oot_macro_prauc, 3) if isinstance(rf_oot_macro_prauc, float) else rf_oot_macro_prauc,
                wf_time="N/A",
                wf_mem="N/A",
                wf_f1="N/A",
                wf_prauc="N/A",
            ))
            pd.DataFrame(results).to_csv(out_file, index=False)
        else:
            if args.mode != "temporal": print("Already completed Baseline: RandomForest (166), skipping.")

        if args.mode != "temporal" and "Baseline: Logistic Regression (166)" not in completed_sweeps:
            from sklearn.linear_model import LogisticRegression
            stat_lr, static_lr_f1, static_lr_prauc = {}, 0.0, 0.0
            if not args.only_wf:
                with profile_resources() as stat_lr:
                    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=cfg_tabular.seed).fit(Xtr_b, ytr_b)
                s_lr = lr.predict_proba(Xte_b)[:, 1]
                static_lr_f1 = f1_score(yte_b, (s_lr >= 0.5).astype(int), pos_label=1)
                static_lr_prauc = average_precision_score(yte_b, s_lr)
                def predict_lr(g_t, m_t):
                    return lr.predict_proba(g_t["x"][m_t, :166].numpy())[:, 1]
                lr_oot_macro_f1, lr_oot_macro_prauc = _evaluate_static_macro(predict_lr, dm_base, cfg_tabular.test_steps)
                if len(cfg_tabular.val_steps) > 0:
                    lr_val_macro_f1, lr_val_macro_prauc = _evaluate_static_macro(predict_lr, dm_base, cfg_tabular.val_steps)
                    if 'Xval_b' not in locals():
                        Xval_b = np.concatenate([dm_base.graphs[t]["x"].numpy()[:, :166] for t in cfg_tabular.val_steps])
                        yval_b = np.concatenate([dm_base.graphs[t]["y"].numpy() for t in cfg_tabular.val_steps])
                        m_val_b = yval_b != -1
                    if m_val_b.sum() > 0:
                        s_val_lr = lr.predict_proba(Xval_b[m_val_b])[:, 1]
                        lr_val_pooled_f1 = f1_score(yval_b[m_val_b], (s_val_lr >= 0.5).astype(int), pos_label=1)
                        lr_val_pooled_prauc = average_precision_score(yval_b[m_val_b], s_val_lr)
                    else:
                        lr_val_pooled_f1, lr_val_pooled_prauc = 0.0, 0.0
                else:
                    lr_val_macro_f1, lr_val_macro_prauc = "N/A", "N/A"
                    lr_val_pooled_f1, lr_val_pooled_prauc = "N/A", "N/A"
                
            results.append(_make_result(
                seed=cfg_tabular.seed, variation="Base", sweep="Baseline: Logistic Regression (166)",
                static_time=round(stat_lr.get("time", 0.0), 3), static_mem=round(stat_lr.get("peak_mem", 0.0), 2),
                static_val_pooled_f1=round(lr_val_pooled_f1, 3) if isinstance(lr_val_pooled_f1, float) else lr_val_pooled_f1,
                static_val_pooled_prauc=round(lr_val_pooled_prauc, 3) if isinstance(lr_val_pooled_prauc, float) else lr_val_pooled_prauc,
                static_val_macro_f1=round(lr_val_macro_f1, 3) if isinstance(lr_val_macro_f1, float) else lr_val_macro_f1,
                static_val_macro_prauc=round(lr_val_macro_prauc, 3) if isinstance(lr_val_macro_prauc, float) else lr_val_macro_prauc,
                static_oot_pooled_f1=round(static_lr_f1, 3), static_oot_pooled_prauc=round(static_lr_prauc, 3),
                static_oot_macro_f1=round(lr_oot_macro_f1, 3) if isinstance(lr_oot_macro_f1, float) else lr_oot_macro_f1,
                static_oot_macro_prauc=round(lr_oot_macro_prauc, 3) if isinstance(lr_oot_macro_prauc, float) else lr_oot_macro_prauc,
                wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
            ))
            pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
        else:
            if args.mode != "temporal": print("Already completed Baseline: Logistic Regression (166), skipping.")

        if args.mode != "temporal" and "Baseline: PyG GCN (2-layer)" not in completed_sweeps:
            import torch.nn as nn
            from torch_geometric.nn import GCNConv
            from torch_geometric.utils import to_undirected
            
            class GCN2(nn.Module):
                def __init__(self, in_dim, hidden, n_classes=2):
                    super().__init__()
                    self.c1 = GCNConv(in_dim, hidden)
                    self.c2 = GCNConv(hidden, n_classes)
                def forward(self, x, edge_index):
                    return self.c2(torch.relu(self.c1(x, edge_index)), edge_index)

            stat_gcn, static_gcn_f1, static_gcn_prauc = {}, 0.0, 0.0
            if not args.only_wf:
                with profile_resources() as stat_gcn:
                    gcn_device = torch.device("cpu")
                    in_dim = dm_base.graphs[min(dm_base.graphs)]["x"].shape[1]
                    model = GCN2(in_dim, 100).to(gcn_device)
                    ytr = torch.cat([dm_base.graphs[t]["y"][dm_base.graphs[t]["labeled_mask"]] for t in cfg_tabular.train_steps if t in dm_base.graphs])
                    cls_w = torch.tensor([0.3, 0.7], dtype=torch.float32, device=gcn_device)
                    loss_fn = nn.CrossEntropyLoss(weight=cls_w)
                    opt = torch.optim.Adam(model.parameters(), lr=0.001)

                    edges = {t: to_undirected(dm_base.graphs[t]["edge_index"]).to(gcn_device) for t in dm_base.graphs}
                    for _ in range(1000):
                        opt.zero_grad()
                        total = 0.0
                        for t in cfg_tabular.train_steps:
                            if t not in dm_base.graphs: continue
                            g = dm_base.graphs[t]; m = g["labeled_mask"]
                            if m.sum() == 0: continue
                            logits = model(g["x"].to(gcn_device), edges[t])
                            total = total + loss_fn(logits[m], g["y"][m].to(gcn_device))
                        total.backward()
                        opt.step()

                    model.eval()
                    s_all, y_all = [], []
                    with torch.no_grad():
                        for t in cfg_tabular.test_steps:
                            if t not in dm_base.graphs: continue
                            g = dm_base.graphs[t]; m = g["labeled_mask"]
                            if m.sum() == 0: continue
                            logits = model(g["x"].to(gcn_device), edges[t])
                            s_all.append(torch.softmax(logits[m], dim=1)[:, 1].cpu().numpy())
                            y_all.append(g["y"][m].numpy())
                    if len(s_all) > 0:
                        y_all_cat = np.concatenate(y_all)
                        s_all_cat = np.concatenate(s_all)
                        static_gcn_f1 = f1_score(y_all_cat, (s_all_cat >= 0.5).astype(int), pos_label=1)
                        static_gcn_prauc = average_precision_score(y_all_cat, s_all_cat)
                        
                    def predict_gcn(g_t, m_t):
                        with torch.no_grad():
                            logits = model(g_t["x"].to(gcn_device), edges[g_t["ts"].item() if "ts" in g_t else min(edges.keys())])
                        return torch.softmax(logits[m_t], dim=1)[:, 1].cpu().numpy()
                    gcn_oot_macro_f1, gcn_oot_macro_prauc = _evaluate_static_macro(predict_gcn, dm_base, cfg_tabular.test_steps)
                    if len(cfg_tabular.val_steps) > 0:
                        gcn_val_macro_f1, gcn_val_macro_prauc = _evaluate_static_macro(predict_gcn, dm_base, cfg_tabular.val_steps)
                        if 'Xval_b' not in locals():
                            yval_b = np.concatenate([dm_base.graphs[t]["y"].numpy() for t in cfg_tabular.val_steps])
                            m_val_b = yval_b != -1
                        if m_val_b.sum() > 0:
                            s_val_gcn_all = []
                            y_val_gcn_all = []
                            with torch.no_grad():
                                for t in cfg_tabular.val_steps:
                                    if t not in dm_base.graphs: continue
                                    g = dm_base.graphs[t]; m = g["labeled_mask"]
                                    if m.sum() == 0: continue
                                    logits = model(g["x"].to(gcn_device), edges[t])
                                    s_val_gcn_all.append(torch.softmax(logits[m], dim=1)[:, 1].cpu().numpy())
                                    y_val_gcn_all.append(g["y"][m].numpy())
                            if len(s_val_gcn_all) > 0:
                                gcn_val_pooled_f1 = f1_score(np.concatenate(y_val_gcn_all), (np.concatenate(s_val_gcn_all) >= 0.5).astype(int), pos_label=1)
                                gcn_val_pooled_prauc = average_precision_score(np.concatenate(y_val_gcn_all), np.concatenate(s_val_gcn_all))
                            else:
                                gcn_val_pooled_f1, gcn_val_pooled_prauc = 0.0, 0.0
                        else:
                            gcn_val_pooled_f1, gcn_val_pooled_prauc = 0.0, 0.0
                    else:
                        gcn_val_macro_f1, gcn_val_macro_prauc = "N/A", "N/A"
                        gcn_val_pooled_f1, gcn_val_pooled_prauc = "N/A", "N/A"
                        
            results.append(_make_result(
                seed=cfg_tabular.seed, variation="Base", sweep="Baseline: PyG GCN (2-layer)",
                static_time=round(stat_gcn.get("time", 0.0), 3), static_mem=round(stat_gcn.get("peak_mem", 0.0), 2),
                static_val_pooled_f1=round(gcn_val_pooled_f1, 3) if isinstance(gcn_val_pooled_f1, float) else gcn_val_pooled_f1,
                static_val_pooled_prauc=round(gcn_val_pooled_prauc, 3) if isinstance(gcn_val_pooled_prauc, float) else gcn_val_pooled_prauc,
                static_val_macro_f1=round(gcn_val_macro_f1, 3) if isinstance(gcn_val_macro_f1, float) else gcn_val_macro_f1,
                static_val_macro_prauc=round(gcn_val_macro_prauc, 3) if isinstance(gcn_val_macro_prauc, float) else gcn_val_macro_prauc,
                static_oot_pooled_f1=round(static_gcn_f1, 3), static_oot_pooled_prauc=round(static_gcn_prauc, 3),
                static_oot_macro_f1=round(gcn_oot_macro_f1, 3) if isinstance(gcn_oot_macro_f1, float) else gcn_oot_macro_f1,
                static_oot_macro_prauc=round(gcn_oot_macro_prauc, 3) if isinstance(gcn_oot_macro_prauc, float) else gcn_oot_macro_prauc,
                wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
            ))
            pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
        else:
            if args.mode != "temporal": print("Already completed Baseline: PyG GCN (2-layer), skipping.")
            
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
        variations = ["Base", "PCA"]
        k_vals = [1, 2, 3]
        dir_vals = [False, True]
        topo_vals = [None, 'late', 'early']
    elif args.mode == "mega_k5":
        seeds = [42, 43, 44]
        variations = ["PCA"]
        k_vals = [5]
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
        
        if var == "PCA":
            from dataclasses import replace
            cfg = replace(cfg, use_pca=True, pca_variance=0.98)
        
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
            
            for k, directional, topo, injection in itertools.product(k_vals, [False, True], [False, True], ['late', 'early']):
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
                    f1_val = res.get("Static_Val_Pooled_F1", 0.0)
                    if pd.notna(f1_val) and isinstance(f1_val, (int, float)) and f1_val > best_grid_f1:
                        best_grid_f1 = f1_val
                        best_grid_key = sweep_key
                        
            print(f"\n  [Phase 2 Winner] {best_grid_key} achieved highest Val F1: {best_grid_f1:.3f}.")
            

            

    # ── PHASE 2.5: Aggregated Champion Selection & Deep Residual Ablation ───────
    import collections
    import re
    config_stats = collections.defaultdict(lambda: {"val_f1": [], "test_f1": []})
    
    for r in results:
        sweep_str = r.get("Sweep", "")
        if not isinstance(sweep_str, str) or not sweep_str.startswith("Grid:"):
            continue
            
        match = re.match(r"^(Grid: .*?) \(Seed \d+, Var (.*?)\)$", sweep_str)
        if match:
            base_name = f"{match.group(1)} (Var {match.group(2)})"
            
            v_f1 = r.get("Static_Val_Macro_F1", 0.0)
            t_f1 = r.get("Static_OOT_Macro_F1", 0.0)
            
            if pd.notna(v_f1) and v_f1 != "N/A":
                config_stats[base_name]["val_f1"].append(float(v_f1))
            if pd.notna(t_f1) and t_f1 != "N/A":
                config_stats[base_name]["test_f1"].append(float(t_f1))

    avg_stats = []
    for base_name, stats in config_stats.items():
        v_avg = sum(stats["val_f1"]) / len(stats["val_f1"]) if stats["val_f1"] else 0.0
        t_avg = sum(stats["test_f1"]) / len(stats["test_f1"]) if stats["test_f1"] else 0.0
        avg_stats.append({'Config': base_name, 'Val_F1': v_avg, 'Test_F1': t_avg})

    df_avg = pd.DataFrame(avg_stats)
    if not df_avg.empty:
        val_75 = df_avg['Val_F1'].quantile(0.75)
        test_75 = df_avg['Test_F1'].quantile(0.75)
        top_both = df_avg[(df_avg['Val_F1'] >= val_75) & (df_avg['Test_F1'] >= test_75)]
        global_champions = top_both['Config'].tolist()
    else:
        global_champions = []
    print(f"\n--- Global Champions Identified ---")
    for champ in global_champions:
        print(f"  - {champ}")



    # ── PHASE 3: Walk-Forward on Global Champions ──────────────────────
    if args.mode != "temporal" and not args.only_static:
        print("\n--- Walk-Forward Validation (Global Champions) ---")
        
        # We define a helper to execute WF for a specific config
        def run_wf_for_config(best_cfg, k, directional, topo_str, var, seed, base_sweep_name):
            best_sweep_name = f"{base_sweep_name} (Seed {seed}, Var {var})"
            wf_name = f"Best WF: {best_sweep_name}"
            
            if wf_name not in completed_sweeps:
                print(f"\nRunning WF on: {best_sweep_name}")
                with profile_resources() as wf_stat:
                    dm_best = EllipticDataModule(df, df_edge, feature_cols, best_cfg)
                    dm_best.setup()
                    _, _, wf_records = walk_forward_validation(dm_best, best_cfg, DEVICE, sweep_name=wf_name, return_records=True)
                    
                from evaluation.wf_metrics import stratified_wf_metrics
                from evaluation.ablation_validation import _write_csv2
                records_dict = [{"tau": r[0], "y_true": r[3], "scores": r[4], "y_pred": r[5]} for r in wf_records]
                agg, rows = stratified_wf_metrics(records_dict, threshold=0.5)
                _write_csv2(wf_name, rows, {})

                wf_res = _make_result(
                    seed=seed,
                    variation=var,
                    sweep=wf_name,
                    static_time="N/A",
                    static_mem="N/A",
                    static_oot_pooled_f1="N/A",
                    static_oot_pooled_prauc="N/A",
                    wf_time=round(wf_stat.get("time", 0.0), 3),
                    wf_mem=round(wf_stat.get("peak_mem", 0.0), 2),
                    wf_f1=round(agg["WF_Macro_F1"], 3),
                    wf_prauc=round(agg["WF_Macro_PRAUC"], 3),
                    wf_pooled_f1=round(agg["WF_Pooled_F1"], 3),
                    wf_pooled_prauc=round(agg["WF_Pooled_PRAUC"], 3),
                    wf_pre43_pooled_f1=round(agg["WF_Pre43_Pooled_F1"], 3),
                    wf_pre43_prauc=round(agg["WF_Pre43_PRAUC"], 3),
                    wf_shock_f1=round(agg["WF_Shock_F1"], 3),
                    wf_shock_prauc=round(agg["WF_Shock_PRAUC"], 3),
                    wf_recovery_pooled_f1=round(agg["WF_Recovery_Pooled_F1"], 3),
                    wf_recovery_prauc=round(agg["WF_Recovery_PRAUC"], 3),
                    feature_set="Prop-N",
                    threshold="τ-1 calibrated",
                )
                results.append(wf_res)
                pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(out_file, index=False)
                print(f"--> {wf_res}\n")
                completed_sweeps.add(wf_name)

        # 1. Run baselines & explicit edge cases for Seed 42
        print("\n--- Running Baseline Sweeps & Edge Cases for Walk-Forward (Seed 42) ---")
        cfg_s1 = Config(use_mlp_head=False, use_multiscale_prop=False, sgc_k=2, use_directional_prop=False, use_graph_structural=False, seed=42)
        run_wf_for_config(cfg_s1, 2, False, 'None', 'Base', 42, "Sweep 1: SGC (baseline)")
        
        cfg_s2 = Config(use_mlp_head=True, use_multiscale_prop=False, sgc_k=2, use_directional_prop=False, use_graph_structural=False, seed=42)
        run_wf_for_config(cfg_s2, 2, False, 'None', 'Base', 42, "Sweep 2: + MLP Head")

        # Edge Case 1: Multi-scale direct comparison to Sweep 2
        cfg_ec1 = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=2, use_directional_prop=False, use_graph_structural=False, seed=42)
        run_wf_for_config(cfg_ec1, 2, False, 'None', 'Base', 42, "Grid: K=2, Dir=F, Topo=None")

        # Edge Case 2 & 3: PCA versions of our winners
        cfg_ec2 = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=1, use_directional_prop=False, use_graph_structural=True, topo_injection_mode='late', use_pca=True, pca_variance=0.98, seed=42)
        run_wf_for_config(cfg_ec2, 1, False, 'late', 'PCA', 42, "Grid: K=1, Dir=F, Topo=late")

        cfg_ec3 = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=3, use_directional_prop=False, use_graph_structural=True, topo_injection_mode='late', use_pca=True, pca_variance=0.98, seed=42)
        run_wf_for_config(cfg_ec3, 3, False, 'late', 'PCA', 42, "Grid: K=3, Dir=F, Topo=late")

        # 2. Run global champions for Seed 42 only
        print("\n--- Running Global Champions for Walk-Forward (Seed 42) ---")
        for champ in global_champions:
            match = re.match(r"^Grid: K=(\d+), Dir=(T|F), Topo=(None|late|early) \(Var (.*?)\)$", champ)
            if not match: continue
            k = int(match.group(1))
            directional = (match.group(2) == 'T')
            topo_str = match.group(3)
            var = match.group(4)
            topo = (topo_str != 'None')
            injection = topo_str if topo else 'late'

            for seed in [42]:
                from dataclasses import replace
                best_cfg = Config(
                    use_mlp_head=True,
                    use_multiscale_prop=True,
                    sgc_k=k,
                    use_directional_prop=directional,
                    use_graph_structural=topo,
                    topo_injection_mode=injection,
                    seed=seed
                )
                if var == "PCA":
                    best_cfg = replace(best_cfg, use_pca=True, pca_variance=0.98)
                
                base_sweep_name = f"Grid: K={k}, Dir={'T' if directional else 'F'}, Topo={topo_str}"
                run_wf_for_config(best_cfg, k, directional, topo_str, var, seed, base_sweep_name)

        # PHASE 4: IPCA Ablation on Global Champions
        print("\n=== PHASE 4: IPCA Ablation ===")
        from evaluation.ablation_validation import evaluate_ipca_wf, evaluate_xgboost_wf
        
        xgb_wf_res = evaluate_xgboost_wf(dm_base, cfg_tabular, _make_result)
        results.append(xgb_wf_res)
        
        pca_champions = [c for c in global_champions if "Var PCA" in c]
        for champ in pca_champions:
            match = re.match(r"^Grid: K=(\d+), Dir=(T|F), Topo=(None|late|early) \(Var (.*?)\)$", champ)
            if not match: continue
            k = int(match.group(1))
            directional = (match.group(2) == 'T')
            topo_str = match.group(3)
            
            cfg_ipca = Config(
                use_mlp_head=True, use_multiscale_prop=True, sgc_k=k,
                use_directional_prop=directional, use_graph_structural=(topo_str != 'None'),
                topo_injection_mode='late' if topo_str != 'None' else 'early',
                use_pca=True, pca_variance=0.98, use_ipca=True, seed=42
            )
            w_name = f"Ablation: IPCA {k} {'T' if directional else 'F'} {topo_str}"
            ipca_dm = EllipticDataModule(df, df_edge, feature_cols, cfg_ipca)
            ipca_dm.setup()
            res = evaluate_ipca_wf(ipca_dm, cfg_ipca, w_name, _make_result)
            results.append(res)
            
        # PHASE 5: Exponential Decay
        print("\n=== PHASE 5: Exponential Decay Ablation ===")
        from evaluation.ablation_validation import evaluate_decay_wf, evaluate_xgb_decay_wf
        
        # 1. Decay on XGBoost
        for lam in [0.05, 0.25, 0.50]:
            xgb_decay_res = evaluate_xgb_decay_wf(dm_base, cfg_tabular, lam, _make_result)
            results.append(xgb_decay_res)

        # 2. Decay on All Global Champions
        for champ in global_champions:
            match = re.match(r"^Grid: K=(\d+), Dir=(T|F), Topo=(None|late|early) \(Var (.*?)\)$", champ)
            if not match: continue
            k = int(match.group(1))
            directional = (match.group(2) == 'T')
            topo_str = match.group(3)
            var = match.group(4)
            
            cfg_decay = Config(
                use_mlp_head=True, use_multiscale_prop=True, sgc_k=k,
                use_directional_prop=directional, use_graph_structural=(topo_str != 'None'),
                topo_injection_mode='late' if topo_str != 'None' else 'early',
                use_pca=(var == "PCA"), pca_variance=0.98 if var == "PCA" else None,
                seed=42
            )
            decay_dm = EllipticDataModule(df, df_edge, feature_cols, cfg_decay)
            decay_dm.setup()
            
            for lam in [0.05, 0.25, 0.50]:
                w_name = f"Ablation: Decay λ={lam} on {k} {'T' if directional else 'F'} {topo_str} {var}"
                res = evaluate_decay_wf(decay_dm, cfg_decay, lam, w_name, _make_result)
                results.append(res)
                    

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
            f"Stat [Time:{str(r['Static_Time_s']):>5s}s, Mem:{str(r['Static_Mem_MB']):>5s}MB] F1={str(r['Static_OOT_Pooled_F1']):<5s} | "
            f"WF [Time:{str(r['WF_Time_s']):>5s}s, Mem:{str(r['WF_Mem_MB']):>5s}MB] F1={str(r['WF_Macro_F1']):<5s}"
        )


if __name__ == "__main__":
    main()
