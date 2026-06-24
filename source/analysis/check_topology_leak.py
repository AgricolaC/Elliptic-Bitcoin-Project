import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(HERE) # Points to 'source' directory
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from data.load_dataset import download_and_load_data
from data.build_graph import reindex_timestep, topological_features

def check_leak():
    print("Loading data...")
    df, edges_df, _, feature_cols = download_and_load_data()
    
    # We'll test on a couple of timesteps
    test_timesteps = [10, 25, 40]
    
    for t in test_timesteps:
        print(f"\n--- Timestep {t} ---")
        sub_df = df[df.ts == t]
        if len(sub_df) == 0: continue
            
        ei, X, y, tx_ids = reindex_timestep(sub_df, edges_df, feature_cols)
        n = len(y)
        
        # Compute topology
        topo_feats = topological_features(ei, n)
        pr = topo_feats[:, 0]
        cl = topo_feats[:, 1]
        
        # Correlate with the 165 raw features
        pr_corrs = []
        cl_corrs = []
        for i, col in enumerate(feature_cols):
            x_col = X[:, i]
            # Handle constant features (std=0)
            if np.std(x_col) == 0:
                pr_corrs.append((col, 0.0))
                cl_corrs.append((col, 0.0))
                continue
                
            r_pr, _ = pearsonr(pr, x_col)
            r_cl, _ = pearsonr(cl, x_col)
            pr_corrs.append((col, abs(r_pr)))
            cl_corrs.append((col, abs(r_cl)))
            
        pr_corrs.sort(key=lambda x: x[1], reverse=True)
        cl_corrs.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 5 Correlated Features with PageRank:")
        for col, r in pr_corrs[:5]:
            print(f"  {col}: {r:.4f}")
            
        print("Top 5 Correlated Features with Clustering Coeff:")
        for col, r in cl_corrs[:5]:
            print(f"  {col}: {r:.4f}")

if __name__ == "__main__":
    check_leak()
