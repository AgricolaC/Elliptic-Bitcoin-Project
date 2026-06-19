import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule

def id_twonn(X, frac=0.9):
    """
    TwoNN intrinsic dimension estimator from nb3b.
    """
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nn.kneighbors(X)
    
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    
    mask = r1 > 0
    r1, r2 = r1[mask], r2[mask]
    
    mu = r2 / r1
    
    # Sort and fit line to empirical CDF
    mu_sorted = np.sort(mu)
    F_emp = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
    
    x_val = np.log(mu_sorted)
    y_val = -np.log(1 - F_emp + 1e-10)
    
    # Keep frac of the points to avoid boundary effects
    keep = int(len(x_val) * frac)
    x_val, y_val = x_val[:keep], y_val[:keep]
    
    d = np.linalg.lstsq(x_val[:, np.newaxis], y_val, rcond=None)[0][0]
    return d

def run():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    t_target = 42
    results = {}
    
    cfg3 = Config(sgc_k=3, use_multiscale_prop=True, use_graph_structural=True)
    dm3 = EllipticDataModule(df, df_edge, feature_cols, cfg3)
    dm3.setup()
    
    prop_42 = dm3.graphs[t_target]["prop"].numpy()
    
    # The actual feature dim includes base (166) + graph_structural (2) = 168
    dim = dm3.feature_dim
    print(f"Feature dim per hop: {dim}")
    
    # We evaluate Intrinsic Dimension (ID) for raw and multiscale combinations
    X_raw = prop_42[:, :dim]
    X_k1 = prop_42[:, :2*dim]
    X_k2 = prop_42[:, :3*dim]
    X_k3 = prop_42[:, :4*dim]
    
    print(f"t={t_target} Nodes: {X_raw.shape[0]}")
    
    id_raw = id_twonn(X_raw)
    id_k1  = id_twonn(X_k1)
    id_k2  = id_twonn(X_k2)
    id_k3  = id_twonn(X_k3)
    
    print("Intrinsic Dimension (TwoNN) at t=42:")
    print(f"Raw Features (K=0): {id_raw:.2f}")
    print(f"Multiscale (K=1): {id_k1:.2f}")
    print(f"Multiscale (K=2): {id_k2:.2f}")
    print(f"Multiscale (K=3): {id_k3:.2f}")

    import pandas as pd
    from config import OUTPUT_DIR
    df_id = pd.DataFrame({
        "Hops (K)": [0, 1, 2, 3],
        "Intrinsic Dimension": [id_raw, id_k1, id_k2, id_k3],
        "F1 Score (Static)": [0.575, 0.586, 0.707, 0.719]
    })
    csv_path = os.path.join(OUTPUT_DIR, "eda_intrinsic_dim.csv")
    df_id.to_csv(csv_path, index=False)
    print(f"Saved intrinsic dimension results to {csv_path}")
        
if __name__ == "__main__":
    run()
