import sys
import os
import time
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elliptic_bitcoin_project"))

from config import Config
from data.load_dataset import download_and_load_data
from run_sweeps import run_static_only_sweep, run_single_sweep

def main():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    cfgs = [
        ("Sweep 4 (Early, Low L2)", Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='early', sgc_weight_decay=5e-4)),
        ("Sweep 4 (Early, High L2)", Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='early', sgc_weight_decay=5e-3)),
        ("Sweep 4 (Late, Low L2)", Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='late', sgc_weight_decay=5e-4)),
        ("Sweep 4 (Late, High L2)", Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=True, topo_injection_mode='late', sgc_weight_decay=5e-3)),
    ]
    
    results = []
    for name, cfg in cfgs:
        print(f"\n==============================================")
        print(f"Running: {name}")
        print(f"==============================================")
        res = run_single_sweep(name, cfg, df, df_edge, feature_cols, window=None)
        results.append(res)
        print(f"Result: {res}")
        
    print("\n\n=== 4-Way Sweep 4 Results ===")
    for r in results:
        print(f"{r['Sweep']:35s} | Stat F1: {r['Static OOT F1']:.3f} | WF F1: {r['Walk-Forward Mean F1']:.3f}")

if __name__ == "__main__":
    main()
