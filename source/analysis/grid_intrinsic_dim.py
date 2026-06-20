import sys
import os
import numpy as np
import pandas as pd
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, set_global_seeds, OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import stack_prop
from sklearn.decomposition import PCA

def id_twonn(X, frac=0.9):
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nn.kneighbors(X)
    
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    
    mask = r1 > 0
    r1, r2 = r1[mask], r2[mask]
    
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    F_emp = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
    
    x_val = np.log(mu_sorted)
    y_val = -np.log(1 - F_emp + 1e-10)
    
    keep = int(len(x_val) * frac)
    x_val, y_val = x_val[:keep], y_val[:keep]
    
    d = np.linalg.lstsq(x_val[:, np.newaxis], y_val, rcond=None)[0][0]
    return d

def run():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    sweep = pd.read_csv(os.path.join(OUTPUT_DIR, "final_aggregated_results.csv"))
    grid_df = sweep[sweep['Sweep'].str.startswith('Grid: ')].copy()
    
    results = []
    
    t_target = 42
    
    for idx, row in grid_df.iterrows():
        sweep_name = row['Sweep']
        variation = row['Variation']
        static_f1 = row['Static_OOT_F1_mean']
        if pd.isna(static_f1):
            static_f1 = row.get('Static_OOT_F1', np.nan)
            
        # Parse configs
        k_match = re.search(r'K=(\d+)', sweep_name)
        dir_match = re.search(r'Dir=([TF])', sweep_name)
        topo_match = re.search(r'Topo=(None|early|late)', sweep_name)
        
        if not k_match or not dir_match or not topo_match:
            continue
            
        k = int(k_match.group(1))
        d_val = dir_match.group(1) == 'T'
        t_val = topo_match.group(1)
        
        use_graph_structural = (t_val != 'None')
        topo_mode = t_val if t_val != 'None' else None
        
        print(f"Evaluating {sweep_name} [{variation}]...")
        cfg = Config(
            sgc_k=k, 
            use_multiscale_prop=True, 
            use_directional_prop=d_val,
            use_graph_structural=use_graph_structural,
            topo_injection_mode=topo_mode,
            seed=42
        )
        set_global_seeds(cfg.seed)
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()
        
        # We extract target snapshot features
        X_target = dm.graphs[t_target]["prop"].numpy()
        
        if variation == 'PCA':
            # Fit PCA on training data [1...34] just as the sweep did
            Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
            pca = PCA(n_components=0.95, random_state=cfg.seed)
            pca.fit(Xtr_g.numpy())
            X_target = pca.transform(X_target)
            
        # Compute TwoNN ID
        computed_id = id_twonn(X_target)
        print(f" -> Intrinsic Dim: {computed_id:.2f} | F1: {static_f1:.3f}")
        
        results.append({
            "Sweep": sweep_name,
            "Variation": variation,
            "K": k,
            "Directional": d_val,
            "Topo": t_val,
            "PCA": variation == 'PCA',
            "Intrinsic Dimension": computed_id,
            "F1 Score": static_f1
        })

    df_res = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "eda_grid_intrinsic_dim.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_res)} configurations to {out_path}")

if __name__ == "__main__":
    run()
