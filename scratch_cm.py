import sys, os
import pandas as pd
import numpy as np
import torch
import joblib

sys.path.append('source')
from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import walk_forward_validation

def main():
    set_global_seeds(42)
    df, df_edge, _, feature_cols = download_and_load_data()
    
    champion_name = "Grid: K=3, Dir=F, Topo=early"
    safe_name = champion_name.replace(" ", "_").replace(":", "").replace(",", "")
    cfg = joblib.load(os.path.join(OUTPUT_DIR, "models", f"{safe_name}_cfg.pkl"))
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    print("Running Walk-Forward for SGC to get records...")
    _, _, sgc_records = walk_forward_validation(dm, cfg, DEVICE, sweep_name=f"Best WF: {champion_name}", return_records=True)
    
    print("Running Walk-Forward for TempXGB to get records...")
    cfg_xgb = Config(train_steps=range(1,27), val_steps=range(27,35), test_steps=range(35,50))
    # TempXGB uses 166 + temporal features, let's just look at SGC records first, we can't easily reproduce TempXGB records here without rewriting the XGB loop.
    # Actually, let's just print SGC confusion matrix at tau >= 43.
    
    total_fp = 0
    total_fn = 0
    total_tp = 0
    
    for tau, step_f1, step_prauc, yte_w, s in sgc_records:
        if tau >= 43:
            threshold = 0.5 # Wait, validation computes threshold via _find_best_f1_threshold, we don't have it explicitly unless we recompute it.
            # But we can approximate by calculating the best threshold or just looking at the raw scores.
            pass
            
    # Instead of re-running TempXGB, I'll just load its predictions from somewhere?
    # No, walk_forward_timesteps.csv only has F1.
    
if __name__ == "__main__":
    pass
