import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import os
from config import Config, OUTPUT_DIR

def plot_temporal_distribution(df, cfg: Config) -> None:
    """Phase 1: Plot nodes, illicit nodes, and illicit fraction over time."""
    n_illicit = len(df[df.label == 1])
    print(f"Illicit fraction overall: {n_illicit/len(df):.4f}  → accuracy is an invalid metric.")
    print(f"Train nodes (t≤34): {(df.ts<=34).sum():,} | Test nodes (t≥35): {(df.ts>=35).sum():,}")

    nodes_per_ts   = df.groupby("ts").size()
    illicit_per_ts = df[df.label == 1].groupby("ts").size().reindex(range(1, 50), fill_value=0)
    labeled_per_ts = df[df.label != -1].groupby("ts").size().reindex(range(1, 50), fill_value=0)
    illicit_frac   = (illicit_per_ts / labeled_per_ts.replace(0, np.nan)).fillna(0)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].bar(nodes_per_ts.index, nodes_per_ts.values, color="#4C72B0")
    ax[0].axvspan(34.5, 49.5, alpha=0.12, color="red")
    ax[0].set_title("Nodes per time step")
    ax[0].set_xlabel("t")

    ax[1].bar(illicit_per_ts.index, illicit_per_ts.values, color="#C44E52")
    ax[1].axvline(cfg.disruption_step, ls="--", color="k")
    ax[1].set_title("Illicit nodes per time step")
    ax[1].set_xlabel("t")

    ax[2].plot(illicit_frac.index, illicit_frac.values, marker="o", color="#C44E52")
    ax[2].axvline(cfg.disruption_step, ls="--", color="k", label="t=43 disruption")
    ax[2].set_title("Illicit fraction among labeled nodes")
    ax[2].set_xlabel("t")
    ax[2].legend()

    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "eda_temporal_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"EDA Plot saved to {out_path}")
