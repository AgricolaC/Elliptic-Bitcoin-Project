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

def plot_degree_distributions(dm: Any, t: int = 42) -> None:
    g = dm.graphs[t]
    ei = g["edge_index"]
    n = g["n"]
    
    if ei.shape[1] == 0:
        print(f"[t={t}] Graph is completely empty. Max in-degree=0, isolated fraction=1.0")
        return
        
    out_deg = torch.bincount(ei[0], minlength=n).numpy()
    in_deg = torch.bincount(ei[1], minlength=n).numpy()
    
    # Assert degree array length
    assert len(in_deg) == n and len(out_deg) == n, "Degree array length mismatch"
    
    isolated_nodes = np.sum((in_deg == 0) & (out_deg == 0))
    print(f"\n--- Degree Distribution (t={t}) ---")
    print(f"Max in-degree: {in_deg.max()}, Max out-degree: {out_deg.max()}")
    print(f"Mean degree: {in_deg.mean():.2f}")
    print(f"Nodes with in-degree > 100: {(in_deg > 100).sum()}")
    print(f"Fraction isolated nodes: {isolated_nodes/n:.4f}")
    
    # Filter 0s for log-log plot
    in_deg_nz = in_deg[in_deg > 0]
    out_deg_nz = out_deg[out_deg > 0]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(in_deg_nz, bins=np.logspace(0, np.log10(in_deg.max() + 1), 50), color="#4C72B0", alpha=0.8)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_title(f"In-Degree Distribution (t={t})")
    ax[0].set_xlabel("In-Degree")
    ax[0].set_ylabel("Count")
    
    ax[1].hist(out_deg_nz, bins=np.logspace(0, np.log10(out_deg.max() + 1), 50), color="#C44E52", alpha=0.8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_title(f"Out-Degree Distribution (t={t})")
    ax[1].set_xlabel("Out-Degree")
    ax[1].set_ylabel("Count")
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "eda_degree_distribution.png")
    plt.savefig(out_path)
    plt.close()

def report_component_structure(dm: Any) -> None:
    import networkx as nx
    print("\n--- Component Structure (Weakly Connected) ---")
    lcc_fracs = []
    ts_list = sorted(list(dm.graphs.keys()))
    for t in ts_list:
        g = dm.graphs[t]
        edge_list = g["edge_index"].t().tolist()
        nx_g = nx.Graph()  # Undirected projection
        nx_g.add_nodes_from(range(g["n"]))
        nx_g.add_edges_from(edge_list)
        
        components = list(nx.connected_components(nx_g))
        n_comps = len(components)
        if n_comps == 0:
            lcc_size = 0
        else:
            lcc_size = max(len(c) for c in components)
        lcc_frac = lcc_size / max(g["n"], 1)
        lcc_fracs.append(lcc_frac)
        print(f"t={t:02d} | Nodes: {g['n']:5d} | Components: {n_comps:5d} | LCC Size: {lcc_size:5d} | LCC Frac: {lcc_frac:.3f}")
        
    plt.figure(figsize=(10, 4))
    plt.plot(ts_list, lcc_fracs, marker="o", color="#55A868")
    plt.title("Largest Weakly Connected Component Fraction over Time")
    plt.xlabel("Timestep (t)")
    plt.ylabel("LCC Fraction")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    out_path = os.path.join(OUTPUT_DIR, "eda_component_structure.png")
    plt.savefig(out_path)
    plt.close()

def plot_raw_feature_manifold(dm: Any, t: int = 42) -> None:
    from sklearn.decomposition import PCA
    g = dm.graphs[t]
    
    # We explicitly extract exactly 166 features. dm.feature_dim handles topo if appended, but the plan asked for 166.
    # To be perfectly rigorous to the dataset, we take the first 166 columns of the RAW tensor.
    X_raw = g["x"].numpy()[:, :166] 
    y = g["y"].numpy()
    
    mask = (y != -1)
    X_known = X_raw[mask]
    y_known = y[mask]
    
    if len(X_known) == 0:
        print(f"[t={t}] No labeled nodes available for raw feature manifold.")
        return
        
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_known)
    var_exp = pca.explained_variance_ratio_
    
    print(f"\n--- Raw Feature Manifold (t={t}) ---")
    print(f"PCA Variance Explained: {var_exp[0]*100:.1f}% (PC1), {var_exp[1]*100:.1f}% (PC2) | Total: {sum(var_exp)*100:.1f}%")
    
    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(X_2d, y_known)
        print(f"2D PCA Silhouette Score (Separability Proxy): {sil:.3f}")
    except ValueError:
        pass
        
    licit_mask = (y_known == 0)
    illicit_mask = (y_known == 1)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[licit_mask, 0], X_2d[licit_mask, 1], c="#4C72B0", s=10, alpha=0.3, label="Licit")
    plt.scatter(X_2d[illicit_mask, 0], X_2d[illicit_mask, 1], c="#C44E52", s=30, alpha=0.9, edgecolors="white", linewidth=0.5, label="Illicit")
    plt.title(f"Raw Feature Manifold (PCA) - t={t}")
    plt.xlabel(f"PC1 ({var_exp[0]:.1%})")
    plt.ylabel(f"PC2 ({var_exp[1]:.1%})")
    plt.legend()
    out_path = os.path.join(OUTPUT_DIR, "eda_raw_feature_manifold.png")
    plt.savefig(out_path)
    plt.close()

def plot_feature_correlation(dm: Any) -> None:
    from sklearn.decomposition import PCA
    import seaborn as sns
    print("\n--- Feature Correlation Matrix ---")
    
    # Train split only (anti-leakage)
    Xs_tr = []
    for t in dm.cfg.train_steps:
        Xs_tr.append(dm.graphs[t]["x"].numpy()[:, :166])
    X_tr_all = np.vstack(Xs_tr)
    
    # Compute correlation matrix
    corr = np.corrcoef(X_tr_all, rowvar=False)
    
    # Handle NaNs from zero-variance columns in small synthetic datasets
    corr = np.nan_to_num(corr, nan=0.0)
    
    # High correlations
    upper_tri = np.triu(corr, k=1)
    high_corr_count = (np.abs(upper_tri) > 0.9).sum()
    print(f"Number of feature pairs with |corr| > 0.9: {high_corr_count}")
    
    # Effective rank via PCA (95% variance)
    pca = PCA()
    pca.fit(X_tr_all)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    effective_rank = np.argmax(cum_var >= 0.95) + 1
    print(f"Effective Rank (PCA components for 95% variance): {effective_rank} / 166")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.abs(corr), cmap="mako", vmin=0, vmax=1)
    plt.title("Absolute Feature Correlation (Train Split Only)")
    out_path = os.path.join(OUTPUT_DIR, "eda_feature_correlation.png")
    plt.savefig(out_path)
    plt.close()

def plot_discriminative_features(dm: Any, top_k: int = 6) -> None:
    print("\n--- Top Discriminative Features ---")
    Xs_tr, ys_tr = [], []
    for t in dm.cfg.train_steps:
        m = dm.graphs[t]["labeled_mask"].numpy()
        Xs_tr.append(dm.graphs[t]["x"].numpy()[:, :166][m])
        ys_tr.append(dm.graphs[t]["y"].numpy()[m])
    
    X_tr = np.concatenate(Xs_tr)
    y_tr = np.concatenate(ys_tr)
    
    if len(np.unique(y_tr)) < 2:
        print("Not enough classes in training set to compute discriminative features.")
        return
        
    licit = X_tr[y_tr == 0]
    illicit = X_tr[y_tr == 1]
    
    # Univariate proxy: absolute diff in means normalized by pooled std
    eps = 1e-8
    pooled_std = np.std(X_tr, axis=0) + eps
    diffs = np.abs(np.mean(licit, axis=0) - np.mean(illicit, axis=0)) / pooled_std
    
    top_indices = np.argsort(diffs)[::-1][:top_k]
    print("Using univariate proxy (normalized diff in means) for top features.")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, f_idx in enumerate(top_indices):
        data = [licit[:, f_idx], illicit[:, f_idx]]
        axes[idx].boxplot(data, labels=["Licit", "Illicit"], showfliers=False)
        axes[idx].set_title(f"Feature {f_idx} (Score: {diffs[f_idx]:.2f})")
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "eda_discriminative_features.png")
    plt.savefig(out_path)
    plt.close()

def run_full_eda(dm: Any, df: pd.DataFrame, cfg: Config, t: int = 42) -> None:
    print("\n" + "="*50)
    print("Executing Comprehensive EDA Module")
    print("="*50)
    
    plots_generated = []
    
    def attempt(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            plots_generated.append(func.__name__)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            
    attempt(plot_temporal_distribution, df, cfg)
    attempt(plot_degree_distributions, dm, t)
    attempt(report_component_structure, dm)
    attempt(plot_raw_feature_manifold, dm, t)
    attempt(plot_feature_correlation, dm)
    attempt(plot_discriminative_features, dm)
    
    print("\n" + "="*50)
    print("EDA Complete. Generated Plots:")
    for p in plots_generated:
        print(f" - {p}.png")
    print("="*50)
