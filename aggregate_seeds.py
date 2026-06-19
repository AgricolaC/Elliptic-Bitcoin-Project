import os
import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def aggregate_sweeps():
    sweep_path = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    if not os.path.exists(sweep_path):
        print(f"File not found: {sweep_path}")
        return

    df = pd.read_csv(sweep_path, keep_default_na=False)
    # Filter numeric columns for aggregation
    numeric_cols = ["Static_OOT_F1", "Static_OOT_PRAUC", "WF_Pre43_PRAUC", "WF_Recovery_PRAUC", "WF_Pooled_PRAUC"]
    
    # Replace 'N/A' with NaN to safely aggregate
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group by Sweep (which represents the model configuration/test)
    agg_df = df.groupby("Sweep")[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-index columns
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    out_path = os.path.join(OUTPUT_DIR, "final_aggregated_results.csv")
    agg_df.to_csv(out_path, index=False)
    print(f"Aggregated {len(agg_df)} sweeps across {df['Seed'].nunique()} seeds.")
    print(f"Saved to {out_path}")

def aggregate_timesteps():
    ts_path = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
    if not os.path.exists(ts_path):
        return

    df = pd.read_csv(ts_path, keep_default_na=False)
    numeric_cols = ["F1", "PRAUC"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Group by Sweep and Tau
    agg_df = df.groupby(["Sweep", "Tau"])[numeric_cols].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    out_path = os.path.join(OUTPUT_DIR, "final_aggregated_timesteps.csv")
    agg_df.to_csv(out_path, index=False)
    print(f"Saved aggregated timesteps to {out_path}")

if __name__ == "__main__":
    aggregate_sweeps()
    aggregate_timesteps()
