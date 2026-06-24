import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist as _scipy_pdist
import networkx as nx

from config import OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import reindex_timestep


def _median_gamma(pX: np.ndarray, cX: np.ndarray) -> float:
    dists = _scipy_pdist(np.vstack([pX, cX]))
    sigma = np.median(dists)
    return 1.0 / (2.0 * sigma ** 2) if sigma > 0 else 1.0


def _mmd_unbiased(pX: np.ndarray, cX: np.ndarray, gamma: float) -> float:
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(pX, pX, gamma=gamma)
    YY = rbf_kernel(cX, cX, gamma=gamma)
    XY = rbf_kernel(pX, cX, gamma=gamma)
    n, m = len(pX), len(cX)
    np.fill_diagonal(XX, 0.0)
    np.fill_diagonal(YY, 0.0)
    return XX.sum() / (n * (n - 1)) + YY.sum() / (m * (m - 1)) - 2 * XY.mean()


def compute_pagerank(edge_index, n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_index.t().tolist())
    pr = nx.pagerank(G, alpha=0.85, max_iter=200) if G.number_of_edges() else {i: 1.0/n for i in range(n)}
    return np.array([pr.get(i, 0.0) for i in range(n)])

def main():
    print("Loading raw dataset...")
    df, df_edge, feat_dim, feature_cols = download_and_load_data()
    
    # 1. PageRank for all labeled nodes
    print("Computing PageRank scores...")
    pr_rows = []
    for t in df.ts.unique():
        sub = df[df.ts == t]
        if len(sub) == 0: continue
        ei, X, y, txids = reindex_timestep(sub, df_edge, feature_cols)
        pr_scores = compute_pagerank(ei, len(y))
        
        # Keep only labeled nodes (0 = licit, 1 = illicit)
        mask = y != -1
        y_lab = y[mask]
        pr_lab = pr_scores[mask]
        for label, score in zip(y_lab, pr_lab):
            pr_rows.append({"label": label, "pagerank": score})
            
    df_pr = pd.DataFrame(pr_rows)
    pr_csv = os.path.join(OUTPUT_DIR, "eda_pagerank.csv")
    df_pr.to_csv(pr_csv, index=False)
    print(f"Saved {len(df_pr)} PageRank scores to {pr_csv}")
    
    # 2. PCA and TSNE for specific snapshots
    target_ts = [1, 42, 43, 44, 49]
    print(f"Computing PCA and TSNE 2D for snapshots {target_ts}...")
    pca_rows = []
    tsne_rows = []
    for t in target_ts:
        sub = df[df.ts == t]
        if len(sub) == 0: continue
        ei, X, y, txids = reindex_timestep(sub, df_edge, feature_cols)
        
        # Keep only labeled nodes to avoid O(N^2) TSNE on 40k unlabeled nodes
        mask = y != -1
        y_lab = y[mask]
        X_lab = X[mask]
        
        print(f"  Fitting PCA for tau={t}...")
        pca = PCA(n_components=2)
        X_pca_lab = pca.fit_transform(X_lab)
        
        print(f"  Fitting TSNE for tau={t}...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', n_jobs=-1)
        X_tsne_lab = tsne.fit_transform(X_lab)
        
        for label, cpca, ctsne in zip(y_lab, X_pca_lab, X_tsne_lab):
            pca_rows.append({"tau": t, "label": label, "pca1": cpca[0], "pca2": cpca[1]})
            tsne_rows.append({"tau": t, "label": label, "tsne1": ctsne[0], "tsne2": ctsne[1]})
            
    df_pca = pd.DataFrame(pca_rows)
    pca_csv = os.path.join(OUTPUT_DIR, "eda_pca.csv")
    df_pca.to_csv(pca_csv, index=False)
    print(f"Saved {len(df_pca)} PCA coordinates to {pca_csv}")
    
    df_tsne = pd.DataFrame(tsne_rows)
    tsne_csv = os.path.join(OUTPUT_DIR, "eda_tsne.csv")
    df_tsne.to_csv(tsne_csv, index=False)
    print(f"Saved {len(df_tsne)} TSNE coordinates to {tsne_csv}")

    # 3. Temporal Edge Homophily (Panel D)
    print("Computing Temporal Edge Homophily...")
    homo_rows = []
    for t in df.ts.unique():
        sub = df[df.ts == t]
        if len(sub) == 0: continue
        ei, X, y, txids = reindex_timestep(sub, df_edge, feature_cols)
        
        src_labels = y[ei[0]]
        tgt_labels = y[ei[1]]
        
        ll = ((src_labels == 0) & (tgt_labels == 0)).sum().item()
        ii = ((src_labels == 1) & (tgt_labels == 1)).sum().item()
        il_licit = (((src_labels == 1) & (tgt_labels == 0)) | ((src_labels == 0) & (tgt_labels == 1))).sum().item()
        il_unk = (((src_labels == 1) & (tgt_labels == -1)) | ((src_labels == -1) & (tgt_labels == 1))).sum().item()
        
        homo_rows.append({"tau": t, "licit_licit": ll, "illicit_illicit": ii, "illicit_licit": il_licit, "illicit_unknown": il_unk})
        
    df_homo = pd.DataFrame(homo_rows)
    df_homo.to_csv(os.path.join(OUTPUT_DIR, "eda_homophily.csv"), index=False)

    # 4. Degree Distribution (Panel E)
    print("Computing Degree Distributions...")
    from torch_geometric.utils import degree
    deg_rows = []
    for t in df.ts.unique():
        sub = df[df.ts == t]
        if len(sub) == 0: continue
        ei, X, y, txids = reindex_timestep(sub, df_edge, feature_cols)
        
        n_nodes = len(y)
        in_deg = degree(ei[1], num_nodes=n_nodes).numpy()
        out_deg = degree(ei[0], num_nodes=n_nodes).numpy()
        
        mask = y != -1
        y_lab = y[mask]
        in_lab = in_deg[mask]
        out_lab = out_deg[mask]
        
        for label, indeg, outdeg in zip(y_lab, in_lab, out_lab):
            deg_rows.append({"label": label, "in_degree": indeg, "out_degree": outdeg})
            
    df_deg = pd.DataFrame(deg_rows)
    df_deg.to_csv(os.path.join(OUTPUT_DIR, "eda_degree.csv"), index=False)
    
    # 5. Manifold Drift Diagnostics (Panel F)
    print("Computing Manifold Drift (MMD & ND-Wasserstein)...")
    try:
        from scipy.stats import wasserstein_distance_nd
    except ImportError:
        wasserstein_distance_nd = None
        print("Warning: scipy.stats.wasserstein_distance_nd not available; Wasserstein will be 0.0.")

    # Fit a single global StandardScaler + PCA on reference steps (ts 1–34)
    # so all per-step Wasserstein comparisons share the same eigenbasis.
    REF_MAX_TS = 34
    X_ref_parts = []
    for t_ref in sorted(t for t in df.ts.unique() if t <= REF_MAX_TS):
        sub_ref = df[df.ts == t_ref]
        if len(sub_ref) == 0:
            continue
        _, X_ref_t, _, _ = reindex_timestep(sub_ref, df_edge, feature_cols)
        X_ref_parts.append(X_ref_t)
    X_ref = np.vstack(X_ref_parts)

    global_scaler = StandardScaler().fit(X_ref)
    global_pca = PCA(n_components=3, random_state=42).fit(global_scaler.transform(X_ref))

    rng = np.random.default_rng(42)
    drift_rows = []
    prev_X = None
    prev_X_pca = None

    for t in sorted(df.ts.unique()):
        sub = df[df.ts == t]
        if len(sub) == 0:
            continue
        ei, X, y, txids = reindex_timestep(sub, df_edge, feature_cols)
        X_np = global_scaler.transform(X)       # standardize with global scaler
        X_pca = global_pca.transform(X_np)      # project onto fixed eigenbasis

        if prev_X is not None:
            max_nodes = 250
            idx_prev = rng.choice(len(prev_X), min(len(prev_X), max_nodes), replace=False)
            idx_curr = rng.choice(len(X_np),   min(len(X_np),   max_nodes), replace=False)

            pX     = prev_X[idx_prev]
            cX     = X_np[idx_curr]
            pX_pca = prev_X_pca[idx_prev]
            cX_pca = X_pca[idx_curr]

            gamma = _median_gamma(pX, cX)
            mmd   = _mmd_unbiased(pX, cX, gamma)

            w_dist = wasserstein_distance_nd(pX_pca, cX_pca) if wasserstein_distance_nd is not None else 0.0
            drift_rows.append({"tau": t, "mmd": mmd, "wasserstein_pca": w_dist})

        prev_X     = X_np
        prev_X_pca = X_pca

    df_drift = pd.DataFrame(drift_rows)
    df_drift.to_csv(os.path.join(OUTPUT_DIR, "eda_drift.csv"), index=False)

if __name__ == "__main__":
    main()
