"""F1 — Memoryless Walk-Forward Arena.

Evaluates two core memoryless baselines across the expanding train window [1..τ-2],
threshold calibrated on τ-1 with ε-fallback, test on τ.
One frozen-preprocessing dm (scalers/prop fit on train_steps 1-26); only the head retrains per τ.

  1. Base XGBoost          — raw 165 features, no memory (the tabular bar)
  2. SGC+MLP static        — K=2 multiscale + MLP head, no temporal recurrence

Writes per-τ rows (CSV-2) and one aggregate row per model (CSV-1).
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(os.path.dirname(HERE))
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, _compute_class_weights, _calibrate_threshold
from evaluation.temporal_validation import _train_illicit_rate, _walk_forward_blocks
from evaluation.wf_metrics import stratified_wf_metrics
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")
TS_CSV = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args, _ = parser.parse_known_args()
SEED = args.seed

CSV2_COLS = ["Sweep", "Seed", "Tau", "N_labeled", "N_illicit", "N_licit",
             "Low_Confidence", "Regime", "Train_Window_Size", "Calib_Threshold",
             "Calib_Fallback", "F1", "PRAUC", "Precision", "Recall", "Selfcond_Bug"]


def _tab_block(dm, steps, use_temporal):
    Xs, ys = [], []
    for t in steps:
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        if m.sum() == 0:
            continue
        feat = g["x"].numpy()[m]
        Xs.append(feat); ys.append(g["y"].numpy()[m])
    return np.concatenate(Xs), np.concatenate(ys)

def _tab_step(dm, tau, use_temporal):
    g = dm.graphs[tau]
    m = g["labeled_mask"].numpy()
    feat = g["x"].numpy()[m]
    return feat, g["y"].numpy()[m]

def collect_tabular(dm, cfg, use_temporal, gir):
    recs, extra = [], {}
    for tau in cfg.test_steps:
        tb, cal = _walk_forward_blocks(dm.graphs, tau)
        if not tb:
            continue
        Xtr, ytr = _tab_block(dm, tb, use_temporal)
        if len(np.unique(ytr)) < 2:
            continue
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=spw, eval_metric="logloss",
                              n_jobs=-1, random_state=SEED)
        model.fit(Xtr, ytr)
        Xte, yte = _tab_step(dm, tau, use_temporal)
        if len(yte) == 0:
            continue
        s = model.predict_proba(Xte)[:, 1]
        thr, fb = 0.5, False
        if cal in dm.graphs:
            Xc, yc = _tab_step(dm, cal, use_temporal)
            if len(yc) > 0 and len(np.unique(yc)) >= 2:
                sc = model.predict_proba(Xc)[:, 1]
                thr, fb = _calibrate_threshold(yc, sc, gir)
        recs.append({"tau": tau, "y_true": yte, "scores": s, "y_pred": (s >= thr).astype(int)})
        extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4),
                      "Calib_Fallback": bool(fb)}
        print(f"    τ={tau} done", flush=True)
    return recs, extra

def collect_sgc_static(dm, cfg, gir):
    recs, extra = [], {}
    for tau in cfg.test_steps:
        tb, cal = _walk_forward_blocks(dm.graphs, tau)
        if not tb:
            continue
            
        # IPCA DYNAMIC UPDATE
        if getattr(cfg, 'use_ipca', False) and hasattr(dm, 'ipca'):
            if tau - 1 in dm.graphs:
                new_raw = dm.graphs[tau - 1]["prop_raw"]
                old_components = dm.ipca.components_.copy()
                
                dm.ipca.partial_fit(new_raw)
                new_components = dm.ipca.components_
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    torch.tensor(old_components), torch.tensor(new_components)
                )
                mean_abs_sim = cos_sim.abs().mean().item()
                print(f"    [IPCA] τ={tau} | Axis Rotation Cosine Sim: {mean_abs_sim:.4f}")
                
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
            if mc.sum() > 0 and len(np.unique(yc)) >= 2:
                with torch.no_grad():
                    sc = torch.softmax(model(Xc[mc].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                thr, fb = _calibrate_threshold(yc, sc, gir)
        recs.append({"tau": tau, "y_true": yte, "scores": s, "y_pred": (s >= thr).astype(int)})
        extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4),
                      "Calib_Fallback": bool(fb)}
        print(f"    τ={tau} done", flush=True)
    return recs, extra

def _migrate_csv2():
    if not os.path.exists(TS_CSV):
        return
    df = pd.read_csv(TS_CSV, keep_default_na=False)
    if "Selfcond_Bug" in df.columns:
        return
    df = df.rename(columns={"Timestep (tau)": "Tau", "PR-AUC": "PRAUC"})
    df["Sweep"] = df["Sweep"].astype(str) + " [superseded-prefix]"
    for c in CSV2_COLS:
        if c not in df.columns:
            df[c] = "N/A"
    df["Selfcond_Bug"] = "present"
    df = df[CSV2_COLS]
    df.to_csv(TS_CSV, index=False)
    print(f"Migrated {len(df)} pre-fix CSV-2 rows to v2 schema (tagged superseded).")

def _write_csv2(sweep, rows, extra):
    out = []
    for r in rows:
        e = extra.get(r["Tau"], {})
        out.append({
            "Sweep": sweep, "Seed": SEED, "Tau": r["Tau"], "N_labeled": r["N_labeled"],
            "N_illicit": r["N_illicit"], "N_licit": r["N_licit"],
            "Low_Confidence": r["Low_Confidence"], "Regime": r["Regime"],
            "Train_Window_Size": e.get("Train_Window_Size", "N/A"),
            "Calib_Threshold": e.get("Calib_Threshold", "N/A"),
            "Calib_Fallback": e.get("Calib_Fallback", "N/A"),
            "F1": r["F1"], "PRAUC": r["PRAUC"], "Precision": r["Precision"], "Recall": r["Recall"],
            "Selfcond_Bug": "fixed",
        })
    df_new = pd.DataFrame(out, columns=CSV2_COLS)
    df = pd.concat([pd.read_csv(TS_CSV, keep_default_na=False), df_new], ignore_index=True) \
        if os.path.exists(TS_CSV) else df_new
    df.to_csv(TS_CSV, index=False)

def _write_csv1(sweep, agg, wf_time, feature_set, threshold_method, variation="Base"):
    df_new = pd.DataFrame([_make_result(
        seed=SEED, variation=variation, sweep=sweep,
        static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
        wf_time=round(wf_time, 2), wf_mem="N/A",
        wf_f1=agg["WF_Macro_F1"], wf_prauc=agg["WF_Macro_PRAUC"],
        wf_pooled_f1=agg["WF_Pooled_F1"], wf_pooled_prauc=agg["WF_Pooled_PRAUC"],
        wf_pre43_pooled_f1=agg["WF_Pre43_Pooled_F1"], wf_pre43_prauc=agg["WF_Pre43_PRAUC"],
        wf_shock_f1=agg["WF_Shock_F1"], wf_shock_prauc=agg["WF_Shock_PRAUC"],
        wf_recovery_pooled_f1=agg["WF_Recovery_Pooled_F1"], wf_recovery_prauc=agg["WF_Recovery_PRAUC"],
        feature_set=feature_set, threshold_method=threshold_method, selfcond_bug="fixed",
        notes="F1 walk-forward, ε-fallback calib, one-step-ahead",
    )], columns=list(_RESULT_KEYS))
    df = pd.concat([pd.read_csv(SWEEP_CSV, keep_default_na=False), df_new], ignore_index=True)
    df.to_csv(SWEEP_CSV, index=False)

def _run(name, collect_fn, feature_set, threshold_method="epsilon-fallback", variation="Base"):
    print(f"\\n=== {name} ===", flush=True)
    t0 = time.time()
    recs, extra = collect_fn()
    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(name, rows, extra)
    _write_csv1(name, agg, time.time() - t0, feature_set, threshold_method, variation=variation)
    print(f"  {name}: Pre43_PRAUC={agg['WF_Pre43_PRAUC']}  Recovery_PRAUC={agg['WF_Recovery_PRAUC']}  "
          f"Pooled_PRAUC={agg['WF_Pooled_PRAUC']}", flush=True)
    return agg

def run():
    set_global_seeds(SEED)
    print("Loading raw dataset...", flush=True)
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg = Config(train_steps=range(1, 27), val_steps=range(27, 35), test_steps=range(35, 50),
                 sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
                 use_graph_structural=False, use_directional_prop=False, seed=SEED)
    print("Building data module (frozen preprocessing on train 1-26)...", flush=True)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    gir = _train_illicit_rate(dm, cfg)
    _migrate_csv2()

    # a_base = _run("F1: Base XGBoost WF [v2]", lambda: collect_tabular(dm, cfg, False, gir),
    #               "Raw-165 (no ts)")
    # a_sgc = _run("F1: SGC+MLP WF K=2 [Dir=F; Topo=None]", lambda: collect_sgc_static(dm, cfg, gir),
    #              f"SGC K=2 multiscale + MLP ({dm.sgc_input_dim}-dim)")

    print("\\n============================================================")
    print("Executing Walk-Forward Validation for Top 4 Graph Winners...")
    print("============================================================\\n")
    
    winners = [
        {"k": 3, "dir": True, "topo": "None", "pca": True},
    ]
    
    for w in winners:
        w_name = f"F1: SGC+MLP WF K={w['k']} [Dir={'T' if w['dir'] else 'F'}; Topo={w['topo']}; PCA]"
        w_cfg = Config(train_steps=range(1, 27), val_steps=range(27, 35), test_steps=range(35, 50),
                       sgc_k=w['k'], use_multiscale_prop=True, use_mlp_head=True,
                       use_graph_structural=(w['topo'] != "None"),
                       topo_injection_mode=w['topo'] if w['topo'] != "None" else "early",
                       use_directional_prop=w['dir'], use_pca=w['pca'], 
                       use_ipca=w['pca'], seed=SEED)
        
        print(f"\\nBuilding DataModule for {w_name}...")
        w_dm = EllipticDataModule(df, df_edge, feature_cols, w_cfg)
        w_dm.setup()
        w_gir = _train_illicit_rate(w_dm, w_cfg)
        
        _run(w_name, lambda: collect_sgc_static(w_dm, w_cfg, w_gir),
             f"K={w['k']} PCA Multiscale", variation="PCA")

if __name__ == "__main__":
    run()
