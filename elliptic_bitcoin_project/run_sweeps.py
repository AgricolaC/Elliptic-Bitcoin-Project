import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import torch
import warnings
import numpy as np
import os
import pandas as pd
from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from models.layers import sgc_propagate
from evaluation.validation import fit_head, stack_prop, walk_forward_validation
from analysis.eda import plot_temporal_distribution
from models.pu_learning import pu_learning_adjust
from models.drift_adaptation import explicit_drift_adaptation
from models.stacking import stacking_meta_classifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score


warnings.filterwarnings("ignore", category=UserWarning)

def run_single_sweep(name: str, cfg: Config, df, df_edge, feature_cols) -> dict:
    set_global_seeds(cfg.seed)
    
    # 1. Setup DataModule
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    # 2. SGC Propagation
    for t in range(1, 50):
        g = dm.graphs[t]
        g["prop"] = sgc_propagate(g["x"], g["edge_index"], cfg.sgc_k, cfg.use_multiscale_prop)
    dm.sgc_input_dim = dm.graphs[1]["prop"].shape[1]
    
    # 3. Static OOT
    Xtr_g, ytr_g = stack_prop(dm, cfg.train_steps)
    Xte_g, yte_g = stack_prop(dm, cfg.test_steps)
    
    if cfg.class_weighted:
        valid_ytr = ytr_g[ytr_g != -1]
        counts = torch.bincount(valid_ytr, minlength=2).float()
        cls_w = (counts.sum() / (2 * counts)).to(DEVICE)
    else:
        cls_w = torch.ones(2, device=DEVICE)
        
    model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    model.eval()
    with torch.no_grad():
        m = (yte_g != -1)
        scores = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        
    from sklearn.metrics import f1_score, average_precision_score
    y_true = yte_g[m].numpy()
    y_pred = (scores >= 0.5).astype(int)
    static_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    static_prauc = average_precision_score(y_true, scores)
    
    # 4. Walk Forward
    wf_f1, wf_prauc = walk_forward_validation(dm, cfg, DEVICE, cls_w)
    
    # Rename walk_forward_drift.png so they don't overwrite each other if desired, 
    # but we will just let it overwrite and they can check sweep_results.csv
    
    return {
        "Sweep": name,
        "Static F1": round(static_f1, 3),
        "Static PR-AUC": round(static_prauc, 3),
        "WF Mean F1": round(wf_f1, 3),
        "WF Mean PR-AUC": round(wf_prauc, 3)
    }

def main():
    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    print("\n--- Running Phase 1: Exploratory Data Analysis ---")
    cfg_default = Config()
    plot_temporal_distribution(df, cfg_default)
    
    print("\n--- Running Phase 0: Baseline Tabular Models ---")
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
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw, eval_metric="aucpr", random_state=cfg_default.seed, n_jobs=1).fit(Xtr_b, ytr_b)
    s_xgb = xgb.predict_proba(Xte_b)[:, 1]
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=1, random_state=cfg_default.seed).fit(Xtr_b, ytr_b)
    s_rf = rf.predict_proba(Xte_b)[:, 1]
    
    results = [
        {"Sweep": "Baseline: XGBoost (166)", "Static OOT F1": round(f1_score(yte_b, (s_xgb >= 0.5).astype(int), pos_label=1), 3), "Static OOT PR-AUC": round(average_precision_score(yte_b, s_xgb), 3), "Walk-Forward Mean F1": "N/A", "Walk-Forward Mean PR-AUC": "N/A"},
        {"Sweep": "Baseline: RandomForest (166)", "Static OOT F1": round(f1_score(yte_b, (s_rf >= 0.5).astype(int), pos_label=1), 3), "Static OOT PR-AUC": round(average_precision_score(yte_b, s_rf), 3), "Walk-Forward Mean F1": "N/A", "Walk-Forward Mean PR-AUC": "N/A"}
    ]
    
    sweeps = [
        ("1. Default SGC", Config(use_mlp_head=False, use_multiscale_prop=False, use_topology=False, use_recon_error=False)),
        ("2. MLP Head On", Config(use_mlp_head=True, use_multiscale_prop=False, use_topology=False, use_recon_error=False)),
        ("3. Multiscale On", Config(use_mlp_head=True, use_multiscale_prop=True, use_topology=False, use_recon_error=False)),
        ("4. Self-Supervision On", Config(use_mlp_head=True, use_multiscale_prop=True, use_topology=True, use_recon_error=True))
    ]
    
    for name, cfg in sweeps:
        print(f"\n=========================================")
        print(f"Running Sweep: {name}")
        print(f"=========================================")
        res = run_single_sweep(name, cfg, df, df_edge, feature_cols)
        results.append(res)
        print(f"--> {res}\n")
        
    print("\n--- Running Advanced Modules ---")
    cfg_advanced = sweeps[3][1] # Full Self-Supervision
    dm_adv = EllipticDataModule(df, df_edge, feature_cols, cfg_advanced)
    dm_adv.setup()
    for t in range(1, 50):
        g = dm_adv.graphs[t]
        g["prop"] = sgc_propagate(g["x"], g["edge_index"], cfg_advanced.sgc_k, cfg_advanced.use_multiscale_prop)
    dm_adv.sgc_input_dim = dm_adv.graphs[1]["prop"].shape[1]
    
    res_pu = pu_learning_adjust(dm_adv, cfg_advanced)
    results.append(res_pu)
    
    res_drift = explicit_drift_adaptation(dm_adv, cfg_advanced)
    results.append(res_drift)
    
    res_stack = stacking_meta_classifier(dm_adv, cfg_advanced)
    results.append(res_stack)
    
    df_res = pd.DataFrame(results)
    out_file = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    df_res.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")
    
    print("\n--- FINAL ABLATION RESULTS ---")
    for r in results:
        print(f"{r['Sweep']:35s} | Static F1={r.get('Static F1', r.get('Static OOT F1'))} | PR-AUC={r.get('Static PR-AUC', r.get('Static OOT PR-AUC'))} | WF F1={r.get('WF Mean F1', r.get('Walk-Forward Mean F1'))} | WF PR-AUC={r.get('WF Mean PR-AUC', r.get('Walk-Forward Mean PR-AUC'))}")

if __name__ == "__main__":
    main()
