import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
from scipy.sparse.linalg import eigsh
from scipy.stats import gaussian_kde
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import roc_auc_score
from typing import Any
from config import OUTPUT_DIR

try:
    import plotly.graph_objects as go
except ImportError:
    pass

def visualize_manifold(dm: Any, slice_t: int = 42, emb_dim: int = 3) -> None:
    g = dm.graphs[slice_t]
    n_full = g["n"]
    G = nx.Graph()
    G.add_nodes_from(range(n_full))
    G.add_edges_from(g["edge_index"].t().tolist())
    
    # Restrict to LCC
    lcc = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc).copy()
    nodes = list(H.nodes())
    n = len(nodes)
    assert n > emb_dim + 1, f"LCC too small ({n}) for a {emb_dim}-D embedding"
    
    L = nx.normalized_laplacian_matrix(H, nodelist=nodes).astype(float)
    assert L.shape == (n, n)
    k = min(emb_dim + 2, n - 1)
    
    try:
        vals, vecs = eigsh(L, k=k, sigma=1e-6, which="LM")
        solver = "shift-invert(σ=1e-6)"
    except Exception as e:
        vals, vecs = eigsh(L + 1e-6 * sp.eye(n), k=k, which="SA")
        solver = f"SA-fallback({type(e).__name__})"
        
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    
    n_trivial = int((vals < 1e-8).sum())
    emb = vecs[:, n_trivial:n_trivial + emb_dim]
    assert emb.shape == (n, emb_dim)
    
    y_local = g["y"].numpy()
    y_lcc = np.array([y_local[nd] for nd in nodes])
    
    print(f"Slice t={slice_t}: full nodes={n_full} | LCC nodes={n} | solver={solver}")
    
    k_nbr = min(20, n - 1)
    
    # 1. KDE
    fit_idx = np.random.choice(n, min(4000, n), replace=False)
    kde = gaussian_kde(emb[fit_idx].T)
    s_kde = -np.log(kde(emb.T) + 1e-12)
    
    # 2. k-NN mean dist
    nn = NearestNeighbors(n_neighbors=k_nbr + 1).fit(emb)
    dist, _ = nn.kneighbors(emb)
    s_knn = dist[:, 1:].mean(axis=1)
    
    # 3. LOF
    lof = LocalOutlierFactor(n_neighbors=k_nbr)
    lof.fit(emb)
    s_lof = -lof.negative_outlier_factor_
    
    scores = {"KDE": s_kde, "kNN-dist": s_knn, "LOF": s_lof}
    
    labeled = y_lcc != -1
    y_eval = (y_lcc[labeled] == 1).astype(int)
    aucs = {name: roc_auc_score(y_eval, s[labeled]) for name, s in scores.items()} \
           if y_eval.sum() else {name: float("nan") for name in scores}
           
    print("── H1 estimator bake-off: void-detection AUC (unsupervised) ─────────")
    for name, a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:9s} AUC = {a:.3f}")
    best = max(aucs, key=aucs.get)
    anomaly = scores[best]
    auc = aucs[best]
    print(f"  → primary estimator = {best}")
    
    m_unk, m_lic, m_ill = (y_lcc == -1), (y_lcc == 0), (y_lcc == 1)
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=emb[m_unk,0], y=emb[m_unk,1], z=emb[m_unk,2], mode="markers",
            name=f"Unknown ({m_unk.sum()})", marker=dict(size=1.4, color="#777", opacity=0.06)))
        fig.add_trace(go.Scatter3d(x=emb[m_lic,0], y=emb[m_lic,1], z=emb[m_lic,2], mode="markers",
            name=f"Licit core ({m_lic.sum()})", marker=dict(size=2.2, color="#4C72B0", opacity=0.18)))
        fig.add_trace(go.Scatter3d(x=emb[m_ill,0], y=emb[m_ill,1], z=emb[m_ill,2], mode="markers",
            name=f"Illicit ({m_ill.sum()})",
            marker=dict(size=5.5, opacity=0.95, color=anomaly[m_ill], colorscale="Inferno",
                        line=dict(width=0.5, color="white"),
                        colorbar=dict(title=f"isolation<br>({best})", x=1.02))))
        fig.update_layout(
            title=f"Spectral forensics t={slice_t} | estimator={best} | void-AUC={auc:.3f}",
            scene=dict(xaxis_title="v₁", yaxis_title="v₂", zaxis_title="v₃", bgcolor="rgba(8,8,18,1)"),
            paper_bgcolor="rgba(8,8,18,1)", font=dict(color="#ddd"),
            legend=dict(itemsizing="constant"), width=900, height=700)
        
        out_file = os.path.join(OUTPUT_DIR, f"spectral_forensics_t{slice_t}.html")
        fig.write_html(out_file)
        print(f"Plot saved to {out_file}")
    except NameError:
        print("Plotly not installed, skipping 3D render.")
