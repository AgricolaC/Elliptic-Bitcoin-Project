import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elliptic_bitcoin_project"))

from config import Config
from data.load_dataset import download_and_load_data
from run_sweeps import run_static_only_sweep

def main():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    cfgs = [
        ("Sweep 3 (PCA 99%)", Config(use_mlp_head=True, use_multiscale_prop=True, use_pca=True, pca_variance=0.99)),
        ("Sweep 4 (PCA 99%)", Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='late', use_pca=True, pca_variance=0.99)),
        ("Sweep 5 (PCA 99%)", Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=True, use_directional_prop=True, topo_injection_mode='early', sgc_weight_decay=5e-3, use_pca=True, pca_variance=0.99)),
    ]
    
    results = []
    for name, cfg in cfgs:
        print(f"\n==============================================")
        print(f"Running: {name}")
        print(f"==============================================")
        res = run_static_only_sweep(name, cfg, df, df_edge, feature_cols)
        results.append(res)
        print(f"Result: {res}")
        
    print("\n\n=== PCA Ablation Results ===")
    for r in results:
        print(f"{r['Sweep']:25s} | F1: {r['Static OOT F1']:.3f} | PR-AUC: {r['Static OOT PR-AUC']:.3f}")

    # Save to CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pca_ablation_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()
