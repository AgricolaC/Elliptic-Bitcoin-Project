import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

from config import OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import reindex_timestep

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
        
        print(f"  Fitting PCA for tau={t}...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        print(f"  Fitting TSNE for tau={t}...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Keep only labeled nodes
        mask = y != -1
        y_lab = y[mask]
        X_pca_lab = X_pca[mask]
        X_tsne_lab = X_tsne[mask]
        
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

if __name__ == "__main__":
    main()
