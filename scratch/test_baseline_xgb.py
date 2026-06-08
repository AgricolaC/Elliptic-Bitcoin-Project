import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_OK"] = "True"
import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elliptic_bitcoin_project"))

from config import Config
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule

def main():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    cfg = Config()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    print("\nStarting Walk-Forward Validation for RAW XGBOOST (Baseline)...", flush=True)
    
    y_true_all = []
    y_pred_all = []
    
    for tau in cfg.test_steps:
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
            
        g_tau = dm.graphs[tau]
        m_tau = g_tau["labeled_mask"].numpy()
        if m_tau.sum() == 0:
            continue
            
        Xte = g_tau["x"].numpy()[:, :166][m_tau]
        yte = g_tau["y"].numpy()[m_tau]
        
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.1, 
            scale_pos_weight=spw, 
            eval_metric="aucpr", 
            random_state=42, 
            n_jobs=1
        ).fit(Xtr, ytr)
        
        s = model.predict_proba(Xte)[:, 1]
        y_pred = (s >= 0.5).astype(int)
        
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        
        step_f1 = f1_score(yte, y_pred, pos_label=1, zero_division=0)
        print(f"  Step {tau:02d} F1: {step_f1:.3f}", flush=True)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    pooled_f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    
    print(f"\n==============================================")
    print(f"Baseline XGBoost Pooled F1: {pooled_f1:.3f}")
    print(f"==============================================")

if __name__ == "__main__":
    main()
