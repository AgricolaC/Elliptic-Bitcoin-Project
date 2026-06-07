import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import Config, set_global_seeds, DEVICE
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import stack_prop, fit_head

def main():
    print(f"torch={torch.__version__} | device={DEVICE}")
    print("\n--- Loading Dataset ---")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    # Base champion config
    base_cfg = Config(use_topology=True, use_multiscale_prop=True, use_mlp_head=True)
    
    print("\n--- Building Propagation Features (Once) ---")
    dm = EllipticDataModule(df, df_edge, feature_cols, base_cfg)
    dm.setup()
    
    seeds = [43, 44]
    heads = {
        "Large": (512, 256, 128)
    }
    
    results = {h: {"train_f1": [], "test_f1": [], "time": [], "params": 0} for h in heads}
    
    for head_name, hidden_dims in heads.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {head_name} Head: {hidden_dims}")
        print(f"{'='*50}")
        
        for s in seeds:
            # 1. Strict Isolation: Only change the seed and mlp_hidden
            cfg = Config(use_topology=True, use_multiscale_prop=True, use_mlp_head=True)
            cfg.seed = s
            cfg.mlp_hidden = hidden_dims
            set_global_seeds(cfg.seed)
            
            # Stack data
            Xtr_g, ytr_g = stack_prop(dm, cfg.train_steps)
            Xte_g, yte_g = stack_prop(dm, cfg.test_steps)
            
            valid_ytr = ytr_g[ytr_g != -1]
            counts = torch.bincount(valid_ytr, minlength=2).float()
            cls_w = (counts.sum() / (2 * counts)).to(DEVICE)
            
            # 2. Train and Time exactly
            t0 = time.perf_counter()
            model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
            train_time = time.perf_counter() - t0
            
            results[head_name]["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results[head_name]["time"].append(train_time)
            
            # 3. Evaluate
            model.eval()
            with torch.no_grad():
                # Train F1 (for overfit detection)
                m_tr = (ytr_g != -1)
                scores_tr = torch.softmax(model(Xtr_g[m_tr].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                f1_tr = f1_score(ytr_g[m_tr].numpy(), (scores_tr >= 0.5).astype(int), pos_label=1)
                
                # Test F1
                m_te = (yte_g != -1)
                scores_te = torch.softmax(model(Xte_g[m_te].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                f1_te = f1_score(yte_g[m_te].numpy(), (scores_te >= 0.5).astype(int), pos_label=1)
                
            results[head_name]["train_f1"].append(f1_tr)
            results[head_name]["test_f1"].append(f1_te)
            
            print(f"  Seed {s:2d} | Train F1: {f1_tr:.3f} | Test F1: {f1_te:.3f} | Time: {train_time:.2f}s")
            
    print(f"\n{'='*50}")
    print("FINAL RESULTS (Mean ± Std over 5 seeds)")
    print(f"{'='*50}")
    
    for h in heads:
        d = results[h]
        tr_mean, tr_std = np.mean(d["train_f1"]), np.std(d["train_f1"])
        te_mean, te_std = np.mean(d["test_f1"]), np.std(d["test_f1"])
        t_mean, t_std   = np.mean(d["time"]), np.std(d["time"])
        
        print(f"[{h} Head: {heads[h]}]")
        print(f"  Parameters : {d['params']:,}")
        print(f"  Train F1   : {tr_mean:.3f} ± {tr_std:.3f}")
        print(f"  Test F1    : {te_mean:.3f} ± {te_std:.3f}")
        print(f"  Train Time : {t_mean:.2f}s ± {t_std:.2f}s")
        if tr_mean - te_mean > 0.05:
            print(f"  > Train/Test Gap: {tr_mean - te_mean:.3f} (Warning: Potential Overfitting)")
        # Save to CSV
    csv_data = []
    for h in heads:
        d = results[h]
        for idx, s in enumerate(seeds):
            if idx < len(d["train_f1"]):
                csv_data.append({
                    "Head": h,
                    "Hidden_Dims": str(heads[h]),
                    "Seed": s,
                    "Parameters": d["params"],
                    "Train_F1": d["train_f1"][idx],
                    "Test_F1": d["test_f1"][idx],
                    "Time_s": d["time"][idx]
                })
    
    if csv_data:
        import os
        from config import OUTPUT_DIR
        out_csv = os.path.join(OUTPUT_DIR, "head_capacity_results.csv")
        pd.DataFrame(csv_data).to_csv(out_csv, index=False)
        print(f"Results saved to {out_csv}")
    print()

if __name__ == "__main__":
    main()
