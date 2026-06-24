import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, average_precision_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule

def main():
    seed = 42
    set_global_seeds(seed)
    print("Loading Elliptic data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    # XGBoost Baseline Config
    cfg_xgb = Config(use_xgb_head=True, use_mlp_head=False, sgc_k=0, use_multiscale_prop=False, use_graph_structural=False, seed=seed)
    dm_xgb = EllipticDataModule(df, df_edge, feature_cols, cfg_xgb)
    dm_xgb.setup()

    results = []
    
    # Variables for pooling test predictions over the formal test window (tau=35 to 49)
    pool_y_true = []
    pool_s_pred = []
    pool_y_pred = []

    print("Running expanding-window walk-forward validation for XGBoost...")
    
    # We test on each tau from 5 to 49 (we need at least a few timesteps to train on)
    for tau in range(5, 50):
        if tau not in dm_xgb.graphs: continue
        
        # Build Train data: t = 1 to tau - 1
        Xs_tr, ys_tr = [], []
        for t in range(1, tau):
            if t in dm_xgb.graphs:
                g = dm_xgb.graphs[t]
                m = g["labeled_mask"]
                if m.sum() > 0:
                    Xs_tr.append(g["x"][m, :166].numpy())
                    ys_tr.append(g["y"][m].numpy())
                    
        if len(Xs_tr) == 0: continue
        Xtr = np.concatenate(Xs_tr)
        ytr = np.concatenate(ys_tr)
        
        # Check if both classes are present in training data
        if len(np.unique(ytr)) < 2: 
            print(f"Skipping tau={tau} due to missing classes in training data.")
            continue
        
        # Build Test data: t = tau
        g_test = dm_xgb.graphs[tau]
        m_test = g_test["labeled_mask"]
        if m_test.sum() == 0: 
            print(f"Skipping tau={tau} due to no labeled test data.")
            continue
            
        Xte = g_test["x"][m_test, :166].numpy()
        yte = g_test["y"][m_test].numpy()
        
        n_labeled = len(yte)
        n_illicit = int((yte == 1).sum())
        
        ratio = float((ytr == 0).sum()) / max(1, float((ytr == 1).sum()))
        model = XGBClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.1,
            scale_pos_weight=ratio,
            eval_metric='aucpr',
            random_state=seed,
            n_jobs=1
        ).fit(Xtr, ytr)
        
        s = model.predict_proba(Xte)[:, 1]
        y_pred = (s >= 0.5).astype(int)
        
        # If test data has only one class, PR-AUC and F1 might be undefined
        if len(np.unique(yte)) > 1:
            pr_auc = float(average_precision_score(yte, s))
        else:
            pr_auc = np.nan
            
        # We can still compute F1 if there are positives, or if we predict positives and there are none.
        f1_05 = float(f1_score(yte, y_pred, pos_label=1, zero_division=0))
        
        results.append({
            "tau": tau,
            "model": "XGBoost",
            "n_labeled": n_labeled,
            "n_illicit": n_illicit,
            "f1_05": round(f1_05, 4),
            "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
            "threshold_used": 0.5
        })
        
        if 35 <= tau <= 49:
            pool_y_true.append(yte)
            pool_s_pred.append(s)
            pool_y_pred.append(y_pred)
            
    print("\n--- Reconciliation Check ---")
    if pool_y_true:
        y_true_all = np.concatenate(pool_y_true)
        s_pred_all = np.concatenate(pool_s_pred)
        y_pred_all = np.concatenate(pool_y_pred)
        pool_pr_auc = float(average_precision_score(y_true_all, s_pred_all))
        pool_f1 = float(f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0))
        print(f"[t=35-49] Pooled F1={pool_f1:.3f}, Pooled PR-AUC={pool_pr_auc:.3f}")
        
        csv_results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "sweep_results.csv")
        if os.path.exists(csv_results_path):
            sweep_df = pd.read_csv(csv_results_path)
            xgb_row = sweep_df[sweep_df["Sweep"] == "Baseline: XGBoost (166)"]
            if len(xgb_row) > 0:
                sweep_f1 = float(xgb_row["WF_Macro_F1"].values[0])
                assert abs(pool_f1 - sweep_f1) < 0.01, f"Reconciliation failed: pooled F1 {pool_f1:.3f} != sweep F1 {sweep_f1:.3f}"
                print("Reconciliation gate passed: plot pooled F1 matches canonical sweep_results.csv F1.")
            else:
                print("Reconciliation check skipped: XGBoost baseline not found in sweep_results.csv.")
        else:
            print("Reconciliation check skipped: sweep_results.csv not found.")
        print("\n")
        
    df_res = pd.DataFrame(results)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "temporal_drift_per_timestep.csv")
    df_res.to_csv(csv_path, index=False)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Primary axis: PR-AUC and F1
    ax1.plot(df_res["tau"], df_res["pr_auc"], label="XGB PR-AUC", color="#4C72B0", marker="o", linewidth=2.5)
    ax1.plot(df_res["tau"], df_res["f1_05"], label="XGB F1@0.5", color="#C44E52", marker="s", linestyle="--", linewidth=2.0, alpha=0.8)
    
    ax1.set_xlabel("Time Step (τ)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Score (PR-AUC / F1)", fontsize=14, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    
    # Secondary axis: n_illicit
    ax2 = ax1.twinx()
    ax2.bar(df_res["tau"], df_res["n_illicit"], color="gray", alpha=0.3, label="n_illicit count")
    ax2.set_ylabel("Number of Illicit Nodes", fontsize=14, fontweight="bold")
    
    # Vertical lines
    ax1.axvline(x=35, color="black", linestyle="--", linewidth=2, label="Test Boundary (t=35)")
    ax1.axvline(x=43, color="red", linestyle="-.", linewidth=2, label="Reported Shutdown (t=43)")
    
    # Grey out points with < 10 illicit nodes
    for idx, row in df_res.iterrows():
        if row["n_illicit"] < 10:
            ax1.axvspan(row["tau"] - 0.5, row["tau"] + 0.5, color='gray', alpha=0.2)
            
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=12)
    
    plt.title("XGBoost Baseline Temporal Drift (Score vs Time)", fontsize=16, fontweight="bold")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    
    png_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "temporal_drift_comparison.png")
    plt.savefig(png_path)
    plt.close()
    
    print(f"Done. Saved results to:\n- {csv_path}\n- {png_path}\n")

if __name__ == "__main__":
    main()
