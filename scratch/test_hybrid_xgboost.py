import sys
import os
import numpy as np

# Sub-process for XGBoost to avoid macOS OpenMP crash with PyTorch
if len(sys.argv) > 1 and sys.argv[1] == "run_xgb":
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, average_precision_score
    
    print("Starting Walk-Forward Validation for Hybrid SGC-XGBoost...", flush=True)
    data = np.load("hybrid_data.npz", allow_pickle=True)
    
    y_true_all = []
    y_pred_all = []
    y_score_all = []
    step_results = []
    
    # 35 to 49
    test_steps = range(35, 50)
    window_size = 4
    
    for tau in test_steps:
        Xs_tr, ys_tr = [], []
        t_start = max(1, tau - window_size)
        for t in range(t_start, tau):
            if f"x_{t}" in data:
                Xs_tr.append(data[f"x_{t}"])
                ys_tr.append(data[f"y_{t}"])
        
        if len(Xs_tr) == 0:
            continue
            
        Xtr = np.concatenate(Xs_tr)
        ytr = np.concatenate(ys_tr)
        
        if (ytr == 1).sum() == 0 or (ytr == 0).sum() == 0:
            continue
            
        if f"x_{tau}" not in data:
            continue
            
        Xte = data[f"x_{tau}"]
        yte = data[f"y_{tau}"]
        
        ratio = float((ytr == 0).sum()) / max(1, float((ytr == 1).sum()))
        model = XGBClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.1,
            scale_pos_weight=ratio,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=1
        ).fit(Xtr, ytr)
        
        s = model.predict_proba(Xte)[:, 1]
        y_pred = (s >= 0.5).astype(int)
        
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        y_score_all.append(s)
        
        step_f1 = f1_score(yte, y_pred, pos_label=1, zero_division=0)
        print(f"  Step {tau:02d} F1: {step_f1:.3f}", flush=True)
        step_results.append({"Step": tau, "F1": step_f1})

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_score_all = np.concatenate(y_score_all)
    
    pooled_f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    pooled_prauc = average_precision_score(y_true_all, y_score_all)
    
    print(f"\n==============================================")
    print(f"Hybrid SGC-XGBoost Results (Window=4)")
    print(f"==============================================")
    print(f"Pooled F1: {pooled_f1:.3f}")
    print(f"Pooled PR-AUC: {pooled_prauc:.3f}")
    
    # Save to CSV
    import pandas as pd
    res_df = pd.DataFrame(step_results)
    res_df.to_csv("results/hybrid_window4_results.csv", index=False)
    print("Saved step-by-step results to results/hybrid_window4_results.csv")
    
    sys.exit(0)

# Main process: Precompute with PyTorch
import torch
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elliptic_bitcoin_project"))
from config import Config
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule

def main():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    cfg = Config(use_mlp_head=False, use_xgb_head=True, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='late')
    
    print("Building Graph & Pre-computing Structural Features (SGC Propagation)...", flush=True)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    # Extract features to NumPy arrays and save
    save_dict = {}
    for t in range(1, 50):
        if t in dm.graphs:
            g = dm.graphs[t]
            m = g["labeled_mask"].numpy()
            if m.sum() > 0:
                save_dict[f"x_{t}"] = g["prop"].numpy()[m]
                save_dict[f"y_{t}"] = g["y"].numpy()[m]
                
    np.savez("hybrid_data.npz", **save_dict)
    
    # Spawn subprocess
    import subprocess
    subprocess.run([sys.executable, __file__, "run_xgb"])

if __name__ == "__main__":
    main()
