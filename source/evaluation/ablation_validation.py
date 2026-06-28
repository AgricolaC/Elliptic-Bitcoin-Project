import os
import time
import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from sweep import profile_resources
from evaluation.validation import fit_head, stack_prop, _compute_class_weights, _calibrate_threshold
from evaluation.temporal_validation import _walk_forward_blocks
from evaluation.wf_metrics import stratified_wf_metrics
from evaluation.validation import SGCHead

TS_CSV = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
CSV2_COLS = ["Sweep", "Seed", "Tau", "N_labeled", "N_illicit", "N_licit",
             "Low_Confidence", "Regime", "Train_Window_Size", "Calib_Threshold",
             "Calib_Fallback", "F1", "PRAUC", "Precision", "Recall", "Selfcond_Bug",
             "Feature_Set", "SGC_K", "Multiscale_Prop", "Directionality", "Topological_Injection", "Decay_Lambda", "Variation"]

def _write_csv2(sweep, rows, extra, seed=42, hyper_cols=None):
    if hyper_cols is None: hyper_cols = {}
    out = []
    for r in rows:
        # Cast to int to avoid float-key mismatch when extra dict uses int keys
        # but r["Tau"] may be float if it was re-read from a CSV.
        tau_key = int(r["Tau"])
        e = extra.get(tau_key, {})
        row_dict = {
            "Sweep": sweep, "Seed": seed, "Tau": tau_key, "N_labeled": r["N_labeled"],
            "N_illicit": r["N_illicit"], "N_licit": r["N_licit"],
            "Low_Confidence": r["Low_Confidence"], "Regime": r["Regime"],
            "Train_Window_Size": e.get("Train_Window_Size", "N/A"),
            "Calib_Threshold": e.get("Calib_Threshold", "N/A"),
            "Calib_Fallback": e.get("Calib_Fallback", "N/A"),
            "F1": r["F1"], "PRAUC": r["PRAUC"], "Precision": r["Precision"], "Recall": r["Recall"],
            "Selfcond_Bug": "fixed",
        }
        for k in ["Feature_Set", "SGC_K", "Multiscale_Prop", "Directionality", "Topological_Injection", "Decay_Lambda", "Variation"]:
            row_dict[k] = hyper_cols.get(k, "N/A")
        out.append(row_dict)
    df_new = pd.DataFrame(out, columns=CSV2_COLS)
    df = pd.concat([pd.read_csv(TS_CSV, keep_default_na=False), df_new], ignore_index=True) \
        if os.path.exists(TS_CSV) else df_new
    df = df.drop_duplicates(subset=["Sweep", "Seed", "Tau"], keep="last")
    df.to_csv(TS_CSV, index=False)

# ==========================================
# 1. XGBOOST WALK-FORWARD
# ==========================================
def _tab_block(dm, steps):
    Xs, ys = [], []
    for t in steps:
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        if m.sum() == 0:
            continue
        feat = g["x"].numpy()[m]
        Xs.append(feat); ys.append(g["y"].numpy()[m])
    return np.concatenate(Xs), np.concatenate(ys)

def _tab_step(dm, tau):
    g = dm.graphs[tau]
    m = g["labeled_mask"].numpy()
    feat = g["x"].numpy()[m]
    return feat, g["y"].numpy()[m]

def evaluate_xgboost_wf(dm, cfg):
    """Walk-forward XGBoost baseline using [1..τ-2] training window + τ-1 calibration.

    Returns the stratified agg dict. The caller (sweep.py) is responsible for
    building the sweep_results.csv row via _make_result so it can merge in the
    static OOT metrics computed separately.
    """
    print("\n--- Running Baseline XGBoost Walk-Forward ---")
    set_global_seeds(42)
    t0 = time.time()
    recs, extra = [], {}
    with profile_resources() as wf_metrics:
        for tau in cfg.test_steps:
            tb, cal = _walk_forward_blocks(dm.graphs, tau)
            if not tb:
                continue
            Xtr, ytr = _tab_block(dm, tb)
            if len(np.unique(ytr)) < 2:
                continue
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  scale_pos_weight=spw, eval_metric="aucpr",
                                  n_jobs=1, random_state=42)
            model.fit(Xtr, ytr)
            Xte, yte = _tab_step(dm, tau)
            if len(yte) == 0:
                continue
            s = model.predict_proba(Xte)[:, 1]
            thr, fb = 0.5, False
            if cal in dm.graphs:
                Xc, yc = _tab_step(dm, cal)
                if len(yc) > 0:
                    sc = model.predict_proba(Xc)[:, 1]
                    thr, fb = _calibrate_threshold(yc, sc)
            recs.append({"tau": tau, "y_true": yte, "scores": s, "y_pred": (s >= thr).astype(int)})
            extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4),
                          "Calib_Fallback": bool(fb)}
            print(f"    [XGBoost] τ={tau} done", flush=True)

    agg, rows = stratified_wf_metrics(recs)
    _write_csv2("Baseline: XGBoost WF (epsilon-fallback)", rows, extra, seed=42)
    print(f"[XGBoost WF] done in {time.time() - t0:.1f}s | "
          f"Pooled F1={agg['WF_Pooled_F1']:.3f} PRAUC={agg['WF_Pooled_PRAUC']:.3f}")
    return agg

# ==========================================
# 2. IPCA DYNAMIC WALK-FORWARD
# ==========================================
def evaluate_ipca_wf(dm, cfg, w_name, make_result_fn):
    print(f"\n--- Running IPCA Walk-Forward: {w_name} ---")
    set_global_seeds(cfg.seed)
    t0 = time.time()
    recs, extra = [], {}
    
    for tau in cfg.test_steps:
        tb, cal = _walk_forward_blocks(dm.graphs, tau)
        if not tb:
            continue
            
        if getattr(cfg, 'use_ipca', False) and hasattr(dm, 'ipca'):
            if tau - 1 in dm.graphs:
                new_raw = dm.graphs[tau - 1]["prop_raw"]
                dm.ipca.partial_fit(new_raw)
            for t in set(tb + [cal, tau]):
                if t in dm.graphs:
                    prop_pca = dm.ipca.transform(dm.graphs[t]["prop_raw"])
                    dm.graphs[t]["prop"] = torch.tensor(prop_pca, dtype=torch.float32)
                    
        Xtr, ytr = stack_prop(dm, tb)
        if len(np.unique(ytr[ytr != -1])) < 2:
            continue
        cls_w = _compute_class_weights(ytr[ytr != -1], DEVICE)
        model = fit_head(Xtr, ytr, dm.sgc_input_dim, cfg, cls_w, DEVICE)
        model.eval()
        
        Xte, yte_all = stack_prop(dm, [tau]); m = yte_all != -1
        if m.sum() == 0:
            continue
        with torch.no_grad():
            s = torch.softmax(model(Xte[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        yte = yte_all[m].numpy()
        
        thr, fb = 0.5, False
        if cal in dm.graphs:
            Xc, yc_all = stack_prop(dm, [cal]); mc = yc_all != -1
            yc = yc_all[mc].numpy()
            if mc.sum() > 0:
                with torch.no_grad():
                    sc = torch.softmax(model(Xc[mc].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                thr, fb = _calibrate_threshold(yc, sc)
                
        recs.append({"tau": tau, "y_true": yte, "scores": s, "y_pred": (s >= thr).astype(int)})
        extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4), "Calib_Fallback": bool(fb)}
        print(f"    [IPCA] τ={tau} done", flush=True)

    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(w_name, rows, extra, seed=cfg.seed)
    
    res = make_result_fn(
        seed=cfg.seed, variation="PCA", sweep=w_name,
        static_time="N/A", static_mem="N/A", static_oot_pooled_f1="N/A", static_oot_pooled_prauc="N/A",
        wf_time=round(time.time() - t0, 3), wf_mem="N/A",
        wf_f1=agg["WF_Macro_F1"], wf_prauc=agg["WF_Macro_PRAUC"],
        wf_pooled_f1=agg["WF_Pooled_F1"], wf_pooled_prauc=agg["WF_Pooled_PRAUC"],
        wf_pre43_pooled_f1=agg["WF_Pre43_Pooled_F1"], wf_pre43_prauc=agg["WF_Pre43_PRAUC"],
        wf_shock_f1=agg["WF_Shock_F1"], wf_shock_prauc=agg["WF_Shock_PRAUC"],
        wf_recovery_pooled_f1=agg["WF_Recovery_Pooled_F1"], wf_recovery_prauc=agg["WF_Recovery_PRAUC"],
        feature_set=f"IPCA Multiscale ({dm.sgc_input_dim}-dim)", threshold="epsilon-fallback",
    )
    return res

# ==========================================
# 3. EXPONENTIAL DECAY WALK-FORWARD
# ==========================================
def get_batch_weights(batch_snapshots, batch_labels, lambda_decay, tau_max, class_mult):
    delta_t = tau_max - batch_snapshots
    temp_w = torch.exp(-lambda_decay * delta_t.float())
    class_w = torch.where(batch_labels == 1, float(class_mult), 1.0)
    combined = temp_w * class_w
    return combined / torch.mean(combined)

def compute_unified_xgb_weights(snapshots, labels, lambda_decay=0.25):
    tau_max = np.max(snapshots)
    delta_t = tau_max - snapshots
    temp_w = np.exp(-lambda_decay * delta_t)
    neg_count = np.sum(labels == 0)
    pos_count = np.sum(labels == 1)
    class_mult = neg_count / max(pos_count, 1)
    class_w = np.where(labels == 1, class_mult, 1.0)
    combined = temp_w * class_w
    return combined / np.mean(combined)

def evaluate_xgb_decay_wf(dm, cfg, lambda_decay, make_result_fn):
    w_name = f"Ablation: Decay λ={lambda_decay} on XGBoost"
    print(f"\n--- Running Decay (λ={lambda_decay}) Walk-Forward: XGBoost ---")
    set_global_seeds(cfg.seed)
    t0 = time.time()
    recs, extra = [], {}

    with profile_resources() as wf_metrics:
        for tau in cfg.test_steps:
            tb, cal = _walk_forward_blocks(dm.graphs, tau)
            if not tb: continue
            Xtr, ytr = _tab_block(dm, tb)
            
            snaps = []
            for t in tb:
                g = dm.graphs[t]; m = g["labeled_mask"].numpy()
                if m.sum() > 0: snaps.extend([t] * int(m.sum()))
            snaps = np.array(snaps)
    
            sample_weights = compute_unified_xgb_weights(snaps, ytr, lambda_decay)
            
            model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  eval_metric="aucpr", random_state=cfg.seed, n_jobs=1)
            model.fit(Xtr, ytr, sample_weight=sample_weights)
    
            Xte, yte = _tab_step(dm, tau)
            if len(yte) == 0: continue
            preds_proba = model.predict_proba(Xte)[:, 1]
    
            thr, fb = 0.5, False
            if cal in dm.graphs:
                Xc, yc = _tab_step(dm, cal)
                if len(yc) > 0:
                    cal_preds = model.predict_proba(Xc)[:, 1]
                    thr, fb = _calibrate_threshold(yc, cal_preds)
                    
            recs.append({"tau": tau, "y_true": yte, "scores": preds_proba, "y_pred": (preds_proba >= thr).astype(int)})
            extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4), "Calib_Fallback": bool(fb)}
            print(f"    [XGB Decay λ={lambda_decay}] τ={tau} done", flush=True)

    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(w_name, rows, extra, seed=cfg.seed)

    res = make_result_fn(
        seed=cfg.seed, variation="Base", sweep=w_name,
        static_time="N/A", static_mem="N/A", static_oot_pooled_f1="N/A", static_oot_pooled_prauc="N/A",
        wf_time=round(time.time() - t0, 3), wf_mem=round(wf_metrics.get("peak_mem", 0.0), 2),
        wf_f1=agg["WF_Macro_F1"], wf_prauc=agg["WF_Macro_PRAUC"],
        wf_pooled_f1=agg["WF_Pooled_F1"], wf_pooled_prauc=agg["WF_Pooled_PRAUC"],
        wf_pre43_pooled_f1=agg["WF_Pre43_Pooled_F1"], wf_pre43_prauc=agg["WF_Pre43_PRAUC"],
        wf_shock_f1=agg["WF_Shock_F1"], wf_shock_prauc=agg["WF_Shock_PRAUC"],
        wf_recovery_pooled_f1=agg["WF_Recovery_Pooled_F1"], wf_recovery_prauc=agg["WF_Recovery_PRAUC"],
        feature_set="Raw-165 (no ts) + exp-decay", threshold="epsilon-fallback",
    )
    return res

def fit_head_decay(Xtr, ytr, snapshots, in_dim, cfg, tau_max, lambda_decay=0.25):
    set_global_seeds(cfg.seed)
    Xtr = Xtr.to(DEVICE)
    ytr = ytr.to(DEVICE)
    snapshots = snapshots.to(DEVICE)
    model = SGCHead(in_dim, cfg).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay)

    mask = (ytr != -1)
    if mask.sum() == 0: return model
    Xtr_m, ytr_m, snaps_m = Xtr[mask], ytr[mask], snapshots[mask]

    neg_count = (ytr_m == 0).sum().item()
    pos_count = (ytr_m == 1).sum().item()
    class_mult = neg_count / max(pos_count, 1)
    batch_w = get_batch_weights(snaps_m, ytr_m, lambda_decay, tau_max, class_mult)

    model.train()
    for _ in range(cfg.sgc_epochs):
        opt.zero_grad()
        logits = model(Xtr_m)
        raw_loss = loss_fn(logits, ytr_m)
        final_loss = torch.mean(raw_loss * batch_w)
        if getattr(cfg, "sgc_l1_lambda", 0.0) > 0.0:
            l1_penalty = model.net[0].weight.abs().sum() if hasattr(model.net[0], 'weight') else 0.0
            final_loss += cfg.sgc_l1_lambda * l1_penalty
        final_loss.backward()
        opt.step()
    return model

def evaluate_decay_wf(dm, cfg, lambda_decay, w_name, make_result_fn):
    print(f"\n--- Running Decay (λ={lambda_decay}) Walk-Forward: {w_name} ---")
    set_global_seeds(cfg.seed)
    t0 = time.time()
    recs, extra = [], {}

    with profile_resources() as wf_metrics:
        for tau in cfg.test_steps:
            tb, cal = _walk_forward_blocks(dm.graphs, tau)
            if not tb:
                continue
            Xs, ys, sn = [], [], []
            for t in tb:
                g = dm.graphs[t]
                Xs.append(g["prop"])
                ys.append(g["y"])
                sn.append(torch.full_like(g["y"], t))
    
            Xtr = torch.cat(Xs)
            ytr = torch.cat(ys)
            snapshots = torch.cat(sn)
    
            model = fit_head_decay(Xtr, ytr, snapshots, in_dim=Xtr.shape[1], cfg=cfg,
                                   tau_max=tau - 1, lambda_decay=lambda_decay)
            model.eval()
    
            Xte = dm.graphs[tau]["prop"].to(DEVICE)
            yte_full = dm.graphs[tau]["y"]
            mask_te = dm.graphs[tau]["labeled_mask"]
            if mask_te.sum() == 0:
                continue
            with torch.no_grad():
                preds_proba = torch.softmax(model(Xte[mask_te]), dim=-1)[:, 1].cpu().numpy()
            yte = yte_full[mask_te].numpy()
    
            thr, fb = 0.5, False
            if cal in dm.graphs:
                Xc = dm.graphs[cal]["prop"].to(DEVICE)
                yc_full = dm.graphs[cal]["y"]
                mask_cal = dm.graphs[cal]["labeled_mask"]
                if mask_cal.sum() > 0:
                    yc = yc_full[mask_cal].numpy()
                    with torch.no_grad():
                        cal_preds = torch.softmax(model(Xc[mask_cal]), dim=-1)[:, 1].cpu().numpy()
                    thr, fb = _calibrate_threshold(yc, cal_preds)
                        
            recs.append({"tau": tau, "y_true": yte, "scores": preds_proba, "y_pred": (preds_proba >= thr).astype(int)})
            extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4), "Calib_Fallback": bool(fb)}
            print(f"    [Decay λ={lambda_decay}] τ={tau} done", flush=True)

    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(w_name, rows, extra, seed=cfg.seed)

    res = make_result_fn(
        seed=cfg.seed, variation="Base", sweep=w_name,
        static_time="N/A", static_mem="N/A", static_oot_pooled_f1="N/A", static_oot_pooled_prauc="N/A",
        wf_time=round(time.time() - t0, 3), wf_mem=round(wf_metrics.get("peak_mem", 0.0), 2),
        wf_f1=agg["WF_Macro_F1"], wf_prauc=agg["WF_Macro_PRAUC"],
        wf_pooled_f1=agg["WF_Pooled_F1"], wf_pooled_prauc=agg["WF_Pooled_PRAUC"],
        wf_pre43_pooled_f1=agg["WF_Pre43_Pooled_F1"], wf_pre43_prauc=agg["WF_Pre43_PRAUC"],
        wf_shock_f1=agg["WF_Shock_F1"], wf_shock_prauc=agg["WF_Shock_PRAUC"],
        wf_recovery_pooled_f1=agg["WF_Recovery_Pooled_F1"], wf_recovery_prauc=agg["WF_Recovery_PRAUC"],
        feature_set=f"Decay Multiscale ({dm.sgc_input_dim}-dim)", threshold="epsilon-fallback",
    )
    return res
