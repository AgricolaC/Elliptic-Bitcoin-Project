import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib

sys.path.append('source')
from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import walk_forward_validation
import networkx as nx

def main():
    set_global_seeds(42)
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    # 1. Run Walk-Forward for SGC Champion
    champion_name = "Grid: K=3, Dir=F, Topo=early"
    safe_name = champion_name.replace(" ", "_").replace(":", "").replace(",", "")
    cfg_path = os.path.join(OUTPUT_DIR, "models", f"{safe_name}_cfg.pkl")
    
    if os.path.exists(cfg_path):
        cfg = joblib.load(cfg_path)
    else:
        cfg = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=3, use_directional_prop=False, use_graph_structural=True, topo_injection_mode="early")
    
    print("Setting up DataModule for SGC...")
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    print(f"Running Walk-Forward for {champion_name}...")
    walk_forward_validation(dm, cfg, DEVICE, sweep_name=f"Best WF: {champion_name}", return_records=False)
    
    # 2. Check the F1 scores around step 43
    print("\n--- F1 Scores around Step 43 ---")
    df_wf = pd.read_csv(os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv"))
    sgc_df = df_wf[df_wf["Sweep"] == f"Best WF: {champion_name}"]
    xgb_df = df_wf[df_wf["Sweep"] == "Baseline: Temporal XGBoost (lag=0)"]
    
    print("SGC Champion F1:")
    for t in [41, 42, 43, 44, 45]:
        val = sgc_df[sgc_df["Timestep (tau)"] == t]["F1"].values
        if len(val): print(f"  tau={t}: {val[0]:.3f}")
        
    print("Temporal XGBoost F1:")
    for t in [41, 42, 43, 44, 45]:
        val = xgb_df[xgb_df["Timestep (tau)"] == t]["F1"].values
        if len(val): print(f"  tau={t}: {val[0]:.3f}")
        
    # 3. Compute topology metrics over time
    print("\n--- Topology Metrics (Edge Density, Mean Degree, Nodes) ---")
    for t in [40, 41, 42, 43, 44, 45, 46]:
        if t in dm.graphs:
            g = dm.graphs[t]
            n = g["x"].size(0)
            e = g["edge_index"].size(1)
            mean_degree = (2 * e) / n if n > 0 else 0
            density = e / (n * (n - 1)) if n > 1 else 0
            illicit = (g["y"] == 1).sum().item()
            print(f"  tau={t}: Nodes={n}, Edges={e}, Illicit={illicit}, MeanDeg={mean_degree:.2f}, Density={density:.6f}")

if __name__ == "__main__":
    main()
