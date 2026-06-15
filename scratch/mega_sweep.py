import sys
import os
import numpy as np
import pandas as pd
import subprocess
import joblib

if len(sys.argv) > 1 and sys.argv[1] == "run_xgb":
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, average_precision_score
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    
    seed = int(sys.argv[2])
    run_wf = sys.argv[3] == "True"
    run_pca = sys.argv[4] == "True"
    run_rf = sys.argv[5] == "True"
    
    data = np.load("mega_data.npz", allow_pickle=True)
    
    # Static OOT
    Xs_tr, ys_tr = [], []
    for t in range(1, 35):
        if f"x_{t}" in data:
            Xs_tr.append(data[f"x_{t}"])
            ys_tr.append(data[f"y_{t}"])
            
    Xtr = np.concatenate(Xs_tr)
    ytr = np.concatenate(ys_tr)
    
    Xs_te, ys_te = [], []
    for t in range(35, 50):
        if f"x_{t}" in data:
            Xs_te.append(data[f"x_{t}"])
            ys_te.append(data[f"y_{t}"])
            
    Xte = np.concatenate(Xs_te)
    yte = np.concatenate(ys_te)
    
    # Feature Selection if enabled
    if run_pca:
        pca = PCA(n_components=0.95, random_state=seed)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
    elif run_rf:
        rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
        rf.fit(Xtr, ytr)
        mask = rf.feature_importances_ > 0.000
        if mask.sum() > 0:
            Xtr = Xtr[:, mask]
            Xte = Xte[:, mask]
    
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
    
    static_f1 = f1_score(yte, y_pred, pos_label=1, zero_division=0)
    static_prauc = average_precision_score(yte, s)
    
    wf_exp_f1 = 0.0
    wf_slide_f1 = 0.0
    
    if run_wf and not run_pca and not run_rf:
        # Expanding
        y_true_all, y_pred_all = [], []
        for tau in range(35, 50):
            Xs_tr_wf, ys_tr_wf = [], []
            for t in range(1, tau):
                if f"x_{t}" in data:
                    Xs_tr_wf.append(data[f"x_{t}"])
                    ys_tr_wf.append(data[f"y_{t}"])
            Xtr_wf = np.concatenate(Xs_tr_wf)
            ytr_wf = np.concatenate(ys_tr_wf)
            
            if f"x_{tau}" not in data: continue
            Xte_wf = data[f"x_{tau}"]
            yte_wf = data[f"y_{tau}"]
            
            ratio_wf = float((ytr_wf == 0).sum()) / max(1, float((ytr_wf == 1).sum()))
            wf_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=ratio_wf, eval_metric='aucpr', random_state=seed, n_jobs=1).fit(Xtr_wf, ytr_wf)
            s_wf = wf_model.predict_proba(Xte_wf)[:, 1]
            y_true_all.append(yte_wf)
            y_pred_all.append((s_wf >= 0.5).astype(int))
            
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        wf_exp_f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
        
        # Sliding
        y_true_all, y_pred_all = [], []
        for tau in range(35, 50):
            Xs_tr_wf, ys_tr_wf = [], []
            t_start = max(1, tau - 5)
            for t in range(t_start, tau):
                if f"x_{t}" in data:
                    Xs_tr_wf.append(data[f"x_{t}"])
                    ys_tr_wf.append(data[f"y_{t}"])
            Xtr_wf = np.concatenate(Xs_tr_wf)
            ytr_wf = np.concatenate(ys_tr_wf)
            
            if f"x_{tau}" not in data: continue
            Xte_wf = data[f"x_{tau}"]
            yte_wf = data[f"y_{tau}"]
            
            ratio_wf = float((ytr_wf == 0).sum()) / max(1, float((ytr_wf == 1).sum()))
            wf_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=ratio_wf, eval_metric='aucpr', random_state=seed, n_jobs=1).fit(Xtr_wf, ytr_wf)
            s_wf = wf_model.predict_proba(Xte_wf)[:, 1]
            y_true_all.append(yte_wf)
            y_pred_all.append((s_wf >= 0.5).astype(int))
            
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        wf_slide_f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
        
    with open("mega_result.txt", "w") as f:
        f.write(f"{static_f1},{static_prauc},{wf_exp_f1},{wf_slide_f1}")
        
    sys.exit(0)

import torch
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elliptic_bitcoin_project"))
from config import Config, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score

DEVICE = torch.device("cpu")

def run_neural(name, cfg, dm, seed, run_pca=False, run_rf=False):
    set_global_seeds(seed)
    Xtr, ytr = stack_prop(dm, list(cfg.train_steps))
    Xte, yte = stack_prop(dm, list(cfg.test_steps))
    
    valid_ytr = ytr[ytr != -1]
    counts = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
    cls_w = (counts.sum() / (2.0 * counts)).to(DEVICE)
    
    m_tr = (ytr != -1)
    m_te = (yte != -1)
    
    if run_pca:
        pca = PCA(n_components=0.95, random_state=seed)
        Xtr_np = pca.fit_transform(Xtr[m_tr].numpy())
        Xte_np = pca.transform(Xte[m_te].numpy())
        Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
        Xte = torch.tensor(Xte_np, dtype=torch.float32)
        ytr = ytr[m_tr]
        yte = yte[m_te]
    elif run_rf:
        rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
        rf.fit(Xtr[m_tr].numpy(), ytr[m_tr].numpy())
        mask = rf.feature_importances_ > 0.000
        if mask.sum() > 0:
            Xtr = Xtr[:, mask]
            Xte = Xte[:, mask]
            
    model = fit_head(Xtr, ytr, Xtr.shape[1], cfg, cls_w, DEVICE, epochs=100) # Fast epochs
    model.eval()
    with torch.no_grad():
        if run_pca:
            scores = torch.softmax(model(Xte.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            y_true = yte.numpy()
        else:
            scores = torch.softmax(model(Xte[m_te].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            y_true = yte[m_te].numpy()
            
    static_f1 = f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)
    static_prauc = average_precision_score(y_true, scores)
    
    return static_f1, static_prauc

def main():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    SEEDS = [42, 43, 44]
    
    configs = [
        ("M1: Baseline XGBoost", Config(use_xgb_head=True, use_mlp_head=False, sgc_k=0, use_multiscale_prop=False, use_graph_structural=False)),
        ("M2: SGC Linear (k=0)", Config(use_mlp_head=False, sgc_k=0, use_multiscale_prop=True, use_graph_structural=False)),
        ("M3: SGC Linear (k=2)", Config(use_mlp_head=False, sgc_k=2, use_multiscale_prop=True, use_graph_structural=False)),
        ("M4: SGC Neural (k=2)", Config(use_mlp_head=True, sgc_k=2, use_multiscale_prop=True, use_graph_structural=False)),
        ("M5: SGC Neural + Topo", Config(use_mlp_head=True, sgc_k=2, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='late')),
        ("M6: SGC Neural + Topo + Dir", Config(use_mlp_head=True, sgc_k=2, use_multiscale_prop=True, use_graph_structural=True, use_directional_prop=True, topo_injection_mode='late')),
        ("M7: Hybrid XGBoost (k=2)", Config(use_xgb_head=True, use_mlp_head=False, sgc_k=2, use_multiscale_prop=True, use_graph_structural=False))
    ]
    
    results = []
    
    for seed in SEEDS:
        print(f"\n{'='*50}\nSEED {seed}\n{'='*50}")
        for name, cfg in configs:
            print(f"Running: {name}")
            set_global_seeds(seed)
            cfg.seed = seed
            dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
            dm.setup()
            
            if cfg.use_xgb_head:
                save_dict = {}
                for t in range(1, 50):
                    if t in dm.graphs:
                        g = dm.graphs[t]
                        m = g["labeled_mask"].numpy()
                        if m.sum() > 0:
                            save_dict[f"x_{t}"] = g["prop"].numpy()[m]
                            save_dict[f"y_{t}"] = g["y"].numpy()[m]
                np.savez("mega_data.npz", **save_dict)
                
                # Base model
                subprocess.run([sys.executable, __file__, "run_xgb", str(seed), "True", "False", "False"])
                with open("mega_result.txt", "r") as f:
                    sf1, sprauc, wfexp, wfslide = map(float, f.read().split(","))
                results.append({"Seed": seed, "Model": name, "Variation": "Base", "Static_F1": sf1, "Static_PRAUC": sprauc, "WF_Expanding_F1": wfexp, "WF_Sliding5_F1": wfslide})
                
                # Ablations for M7
                if name.startswith("M7"):
                    # PCA
                    subprocess.run([sys.executable, __file__, "run_xgb", str(seed), "False", "True", "False"])
                    with open("mega_result.txt", "r") as f:
                        sf1, sprauc, _, _ = map(float, f.read().split(","))
                    results.append({"Seed": seed, "Model": name, "Variation": "PCA", "Static_F1": sf1, "Static_PRAUC": sprauc, "WF_Expanding_F1": 0.0, "WF_Sliding5_F1": 0.0})
                    # RF
                    subprocess.run([sys.executable, __file__, "run_xgb", str(seed), "False", "False", "True"])
                    with open("mega_result.txt", "r") as f:
                        sf1, sprauc, _, _ = map(float, f.read().split(","))
                    results.append({"Seed": seed, "Model": name, "Variation": "RF_Pruned", "Static_F1": sf1, "Static_PRAUC": sprauc, "WF_Expanding_F1": 0.0, "WF_Sliding5_F1": 0.0})
                    
            else:
                # PyTorch
                sf1, sprauc = run_neural(name, cfg, dm, seed)
                results.append({"Seed": seed, "Model": name, "Variation": "Base", "Static_F1": sf1, "Static_PRAUC": sprauc, "WF_Expanding_F1": 0.0, "WF_Sliding5_F1": 0.0})
                
                if name.startswith("M4") or name.startswith("M5") or name.startswith("M6"):
                    # PCA
                    sf1, sprauc = run_neural(name, cfg, dm, seed, run_pca=True)
                    results.append({"Seed": seed, "Model": name, "Variation": "PCA", "Static_F1": sf1, "Static_PRAUC": sprauc, "WF_Expanding_F1": 0.0, "WF_Sliding5_F1": 0.0})
                    # RF
                    sf1, sprauc = run_neural(name, cfg, dm, seed, run_rf=True)
                    results.append({"Seed": seed, "Model": name, "Variation": "RF_Pruned", "Static_F1": sf1, "Static_PRAUC": sprauc, "WF_Expanding_F1": 0.0, "WF_Sliding5_F1": 0.0})
                    
    if os.path.exists("mega_data.npz"): os.remove("mega_data.npz")
    if os.path.exists("mega_result.txt"): os.remove("mega_result.txt")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/mega_sweep_raw.csv", index=False)
    
    # Aggregate Means
    grouped = df_res.groupby(["Model", "Variation"]).agg({
        "Static_F1": ["mean", "std"],
        "WF_Expanding_F1": ["mean", "std"],
        "WF_Sliding5_F1": ["mean", "std"]
    }).round(3)
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.reset_index(inplace=True)
    grouped.to_csv("results/mega_sweep_aggregated.csv", index=False)
    
    print("\n\nMEGA SWEEP COMPLETED. AGGREGATED RESULTS:")
    print(grouped.to_string(index=False))

if __name__ == "__main__":
    main()
