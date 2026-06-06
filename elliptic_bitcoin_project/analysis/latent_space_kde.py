import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from config import Config, set_global_seeds, OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from models.layers import sgc_propagate

def plot_latent_space_kde(slice_t: int = 42) -> None:
    """
    Generate the 'Aha!' moment Latent Space Visualization using UMAP and KDE.
    Projects the un-trained SGC features (multiscale relational topology) into 2D,
    and applies Kernel Density Estimation to show the 'Licit Continent' vs 'Illicit Islands'.
    """
    print("Loading data for KDE Latent Map...")
    cfg = Config()
    set_global_seeds(cfg.seed)
    
    df, df_edge, _, feature_cols = download_and_load_data()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    # We use t=42 (the disruption timestep) as it's the most interesting
    g = dm.graphs[slice_t]
    
    # Use SGC propagated features (relational topology + base features)
    X = g["x"]
    ei = g["edge_index"]
    
    print(f"Propagating SGC features for t={slice_t}...")
    X_prop = sgc_propagate(X, ei, cfg.sgc_k, cfg.use_multiscale_prop).numpy()
    y = g["y"].numpy()
    
    # Filter out unknown nodes (-1)
    mask = (y != -1)
    X_known = X_prop[mask]
    y_known = y[mask]
    
    print(f"Running UMAP projection on {len(y_known)} known nodes...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=cfg.seed)
    X_2d = reducer.fit_transform(X_known)
    
    licit_mask = (y_known == 0)
    illicit_mask = (y_known == 1)
    
    print("Generating KDE Topographic Map...")
    plt.figure(figsize=(12, 10), facecolor="#080812")
    ax = plt.gca()
    ax.set_facecolor("#080812")
    
    # Plot Licit Continent (KDE)
    sns.kdeplot(
        x=X_2d[licit_mask, 0], 
        y=X_2d[licit_mask, 1],
        cmap="Blues", 
        fill=True, 
        alpha=0.6, 
        levels=10, 
        thresh=0.05,
        ax=ax
    )
    
    # Plot Illicit Islands (Scatter + KDE)
    sns.kdeplot(
        x=X_2d[illicit_mask, 0], 
        y=X_2d[illicit_mask, 1],
        cmap="Reds", 
        fill=True, 
        alpha=0.6, 
        levels=6, 
        thresh=0.05,
        ax=ax
    )
    
    # Add scatter for stark contrast
    plt.scatter(
        X_2d[licit_mask, 0], X_2d[licit_mask, 1],
        s=2, c="#4C72B0", alpha=0.1, label="Licit Nodes"
    )
    plt.scatter(
        X_2d[illicit_mask, 0], X_2d[illicit_mask, 1],
        s=15, c="#C44E52", alpha=0.9, edgecolors="white", linewidth=0.5, label="Illicit Nodes"
    )
    
    plt.title(f"Latent Space Topography (SGC + UMAP) - t={slice_t}", color="#dddddd", fontsize=16, pad=20)
    plt.xlabel("UMAP 1", color="#aaaaaa")
    plt.ylabel("UMAP 2", color="#aaaaaa")
    
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_color('#333333')
        
    legend = plt.legend(facecolor="#111111", edgecolor="#333333", labelcolor="#dddddd")
    
    out_file = os.path.join(OUTPUT_DIR, f"latent_space_kde_t{slice_t}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="#080812")
    plt.close()
    
    print(f"Topographic map saved to {out_file}")

if __name__ == "__main__":
    import warnings
    from numba.core.errors import NumbaDeprecationWarning
    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    plot_latent_space_kde()
