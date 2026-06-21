"""
Unified module for temporal analysis, plotting, and seed aggregation.
Merged from analysis_temporal_per_step.py, plot_grid.py, and aggregate_seeds.py.
"""
import sys
import os
import argparse
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import f1_score, average_precision_score

# Ensure source/ is in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(HERE) # get the source/ directory
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from config import Config, DEVICE, OUTPUT_DIR, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.temporal_validation import (
    train_lstm_conditioned, train_ema_conditioned,
    _onestep_blocks, _temporal_state,
)
from evaluation.validation import _find_best_f1_threshold

# ==========================================
# AGGREGATION FUNCTIONS
# ==========================================
def aggregate_sweeps():
    sweep_path = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    if not os.path.exists(sweep_path):
        print(f"File not found: {sweep_path}")
        return

    df = pd.read_csv(sweep_path, keep_default_na=False)
    # Filter numeric columns for aggregation
    numeric_cols = [
        "Static_OOT_F1", "Static_OOT_PRAUC",
        "WF_Pooled_F1", "WF_Pooled_PRAUC",
        "WF_Macro_F1", "WF_Macro_PRAUC",
        "WF_Pre43_Pooled_F1", "WF_Pre43_PRAUC",
        "WF_Shock_F1", "WF_Shock_PRAUC",
        "WF_Recovery_Pooled_F1", "WF_Recovery_PRAUC"
    ]
    
    # Replace 'N/A' with NaN to safely aggregate
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract base sweep name by removing the ' (Seed X, Var Y)' suffix
    df['Base_Sweep'] = df['Sweep'].apply(lambda x: re.sub(r' \(Seed \d+, Var .*\)', '', x) if isinstance(x, str) else x)
    
    # Group by Base_Sweep and Variation
    agg_df = df.groupby(["Base_Sweep", "Variation"])[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-index columns
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    agg_df.rename(columns={'Base_Sweep': 'Sweep'}, inplace=True)
    
    out_path = os.path.join(OUTPUT_DIR, "final_aggregated_results.csv")
    agg_df.to_csv(out_path, index=False)
    print(f"Aggregated {len(agg_df)} sweeps across {df['Seed'].nunique()} seeds.")
    print(f"Saved to {out_path}")

def aggregate_timesteps():
    ts_path = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
    if not os.path.exists(ts_path):
        return

    df = pd.read_csv(ts_path, keep_default_na=False)
    numeric_cols = ["F1", "PR-AUC", "PRAUC"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df['Base_Sweep'] = df['Sweep'].apply(lambda x: re.sub(r' \(Seed \d+, Var .*\)', '', x) if isinstance(x, str) else x)
    
    # Handle schema migration: 'Timestep (tau)' might have been renamed to 'Tau' by _migrate_csv2()
    ts_col = "Tau" if "Tau" in df.columns else "Timestep (tau)"
    
    valid_cols = [c for c in numeric_cols if c in df.columns]
    agg_df = df.groupby(["Base_Sweep", ts_col])[valid_cols].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    agg_df.rename(columns={'Base_Sweep': 'Sweep', ts_col: 'Tau'}, inplace=True)
    if 'PR-AUC_mean' in agg_df.columns:
        agg_df.rename(columns={'PR-AUC_mean': 'PRAUC_mean'}, inplace=True)
    if 'PRAUC_mean' in agg_df.columns:
        agg_df['PRAUC'] = agg_df['PRAUC_mean'] # Fallback for presentation script
    
    out_path = os.path.join(OUTPUT_DIR, "final_aggregated_timesteps.csv")
    agg_df.to_csv(out_path, index=False)
    print(f"Saved aggregated timesteps to {out_path}")


# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_grid_performance():
    RESULTS = os.path.join(OUTPUT_DIR, "final_aggregated_results.csv")
    FIGDIR = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(FIGDIR, exist_ok=True)

    if not os.path.exists(RESULTS):
        print(f"Missing results file {RESULTS}")
        return

    df = pd.read_csv(RESULTS)

    # Filter to only Grid and MLP sweeps, excluding Wide/Residual variations
    df = df[df['Sweep'].str.startswith('Grid:') | df['Sweep'].str.startswith('MLP-')].copy()
    df = df[~df['Sweep'].str.contains('Wide|Residual|ResWide', na=False)]

    def parse_sweep(sweep_str):
        arch = "Linear"
        k = 1
        dir_prop = "F"
        topo = "None"
        
        if sweep_str.startswith("Grid:"):
            arch = "Linear"
            params_str = sweep_str.replace("Grid: ", "")
        else:
            match = re.match(r"MLP-([A-Za-z]+) \[(.*)\]", sweep_str)
            if match:
                arch = match.group(1)
                params_str = match.group(2)
            else:
                params_str = ""
                
        for part in params_str.split(", "):
            if part.startswith("K="):
                k = int(part.split("=")[1])
            elif part.startswith("Dir="):
                dir_prop = part.split("=")[1]
            elif part.startswith("Topo="):
                topo = part.split("=")[1]
                
        return pd.Series([arch, k, dir_prop, topo])

    df[['Arch', 'K', 'Dir', 'Topo']] = df['Sweep'].apply(parse_sweep)

    f1_col = 'Static_OOT_F1_mean' if 'Static_OOT_F1_mean' in df.columns else 'Static_OOT_F1'

    plt.figure(figsize=(16, 10))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    g = sns.catplot(
        data=df, 
        x="K", y=f1_col, 
        hue="Variation", 
        col="Arch", row="Topo", 
        kind="point", 
        markers=["o", "s", "D"], linestyles=["-", "--", "-."],
        height=3, aspect=1.2,
        palette="deep"
    )

    g.set_axis_labels("Propagation Hops (K)", "Static OOT F1 Score")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("Grid Search Performance across Architectures, Topology, and K-Hops", fontsize=16, fontweight='bold')

    out_path = os.path.join(FIGDIR, "grid_performance.png")
    g.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_path}")


# ==========================================
# TEMPORAL WALK-FORWARD
# ==========================================
def _per_step(dm, cfg, device, epochs, embed_dim, kind):
    rows = []
    for tau in cfg.test_steps:
        train_block, calib_step, calib_state, infer_state = _onestep_blocks(dm.graphs, tau)
        if not train_block:
            continue

        g = dm.graphs[tau]
        m = g["labeled_mask"]
        if m.sum() == 0:
            continue
        yte = g["y"][m].numpy()
        if len(np.unique(yte)) < 2:
            continue

        if kind == "lstm":
            embedder, temporal, head = train_lstm_conditioned(
                dm, train_block, cfg, device, epochs=epochs, embed_dim=embed_dim
            )
            embedder.eval(); temporal.eval(); head.eval()
        else:  # ema
            embedder, temporal, head = train_ema_conditioned(
                dm, train_block, cfg, device, epochs=epochs, embed_dim=embed_dim
            )
            embedder.eval(); head.eval()

        # Threshold calibration on tau-1 (filtering state includes tau-1)
        threshold = 0.5
        if calib_step in dm.graphs:
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        h_cal = _temporal_state(embedder, temporal, calib_state, dm, device)
                        logits_cal = head(g_cal["prop"][m_cal].to(device), h_cal)
                        s_cal = torch.softmax(logits_cal, dim=1)[:, 1].cpu().numpy()
                    threshold = _find_best_f1_threshold(y_cal, s_cal)

        # Test on tau (one-step-ahead: state excludes tau)
        with torch.no_grad():
            h_tau = _temporal_state(embedder, temporal, infer_state, dm, device)
            logits_te = head(g["prop"][m].to(device), h_tau)
            s = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()

        y_pred = (s >= threshold).astype(int)
        step_f1 = float(f1_score(yte, y_pred, pos_label=1, zero_division=0))
        step_prauc = float(average_precision_score(yte, s))
        rows.append((tau, step_f1, step_prauc))
        print(f"  [{kind.upper()}] tau={tau:>2d}  F1={step_f1:.3f}  PR-AUC={step_prauc:.3f}", flush=True)
    return rows


def _append_rows(sweep_name, rows):
    csv_file = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
    df = pd.DataFrame(
        {"Sweep": [sweep_name] * len(rows),
         "Timestep (tau)": [r[0] for r in rows],
         "F1": [r[1] for r in rows],
         "PR-AUC": [r[2] for r in rows]}
    )
    header = not os.path.exists(csv_file)
    df.to_csv(csv_file, mode="a", header=header, index=False)
    print(f"Appended {len(rows)} rows for '{sweep_name}' -> {csv_file}", flush=True)


def run_temporal_analysis(epochs=100, embed_dim=32, models="lstm,ema"):
    cfg = Config()  # deep-structural learned-phi representative
    set_global_seeds(cfg.seed)

    print("Loading raw dataset...", flush=True)
    df, df_edge, _, feature_cols = download_and_load_data()

    print("Building data module (SGC propagation + topology)... this is the slow part.", flush=True)
    t0 = time.time()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    print(f"Setup done in {time.time() - t0:.1f}s | sgc_input_dim={dm.sgc_input_dim}", flush=True)

    # LSTM must run on CPU under MPS (parity with sweep.py main())
    lstm_device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE

    wanted = [m.strip() for m in models.split(",") if m.strip()]

    if "ema" in wanted:
        print("\n=== SGC-EMA (learned phi, memoryless-ish baseline) ===", flush=True)
        t0 = time.time()
        rows = _per_step(dm, cfg, lstm_device, epochs, embed_dim, "ema")
        print(f"EMA walk-forward done in {time.time() - t0:.1f}s", flush=True)
        _append_rows("SGC-EMA Conditioned (learned phi)", rows)

    if "lstm" in wanted:
        print("\n=== SGC-LSTM (learned phi, deep structural) ===", flush=True)
        t0 = time.time()
        rows = _per_step(dm, cfg, lstm_device, epochs, embed_dim, "lstm")
        print(f"LSTM walk-forward done in {time.time() - t0:.1f}s", flush=True)
        _append_rows("SGC-LSTM Conditioned (learned phi)", rows)

    print("\nDone.", flush=True)

# ==========================================
# CLI ENTRYPOINT
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="Unified Analysis & Plotting Module")
    ap.add_argument("--action", choices=["aggregate", "plot", "walk-forward"], required=True)
    ap.add_argument("--epochs", type=int, default=100, help="WF epochs per tau")
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--models", type=str, default="lstm,ema", help="comma list: lstm,ema")
    args = ap.parse_args()

    if args.action == "aggregate":
        aggregate_sweeps()
        aggregate_timesteps()
    elif args.action == "plot":
        plot_grid_performance()
    elif args.action == "walk-forward":
        run_temporal_analysis(args.epochs, args.embed_dim, args.models)

if __name__ == "__main__":
    main()
