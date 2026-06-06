import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import torch
import warnings

from config import Config, set_global_seeds, DEVICE
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from models.layers import sgc_propagate
from models.baselines import run_baselines
from evaluation.validation import walk_forward_validation, stack_prop
from analysis.manifold_visualization import visualize_manifold

warnings.filterwarnings("ignore", category=UserWarning)

def main() -> None:
    # 1. Initialize Configuration & Seeds
    cfg = Config()
    set_global_seeds(cfg.seed)
    print(f"torch={torch.__version__} | device={DEVICE} | seed={cfg.seed}")
    
    # 2. Data Loading & Schema Guards
    df, df_edge, node_feature_dim, feature_cols = download_and_load_data()
    print(f"nodes={len(df):,} | edges={len(df_edge):,} | raw_features={node_feature_dim}")
    
    # 3. Graph Building, Scaling & injections
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    # 4. SGC Propagation (Symmetrized Graph inside)
    t0 = time.time()
    for t in range(1, 50):
        g = dm.graphs[t]
        g["prop"] = sgc_propagate(g["x"], g["edge_index"], cfg.sgc_k, cfg.use_multiscale_prop)
        
    dm.sgc_input_dim = dm.graphs[1]["prop"].shape[1]
    print(f"Propagated all 49 slices in {time.time()-t0:.1f}s | SGC input dim = {dm.sgc_input_dim}")
    
    # 5. Baselines (OOT Split on Raw 166 Features)
    print("\n--- Running Tree Baselines ---")
    run_baselines(dm, cfg)
    
    # 6. Neural Network Head Static Setup (OOT Split)
    Xtr_g, ytr_g = stack_prop(dm, cfg.train_steps)
    Xte_g, yte_g = stack_prop(dm, cfg.test_steps)
    
    if cfg.class_weighted:
        # compute weight only on labeled positive/negative (mask out -1)
        valid_ytr = ytr_g[ytr_g != -1]
        counts = torch.bincount(valid_ytr, minlength=2).float()
        cls_w = (counts.sum() / (2 * counts)).to(DEVICE)
    else:
        cls_w = torch.ones(2, device=DEVICE)
        
    from evaluation.validation import fit_head
    
    print("\n--- Training Static SGCHead ---")
    model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    
    model.eval()
    with torch.no_grad():
        # Evaluate statically on valid labeled test nodes
        m = (yte_g != -1)
        scores = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        
    from models.baselines import report
    tag = f"SIGN(K={cfg.sgc_k})" if (cfg.use_multiscale_prop or cfg.use_mlp_head) else f"SGC(K={cfg.sgc_k})"
    print("\n--- Static OOT SGC Comparison ---")
    report(tag, yte_g[m].numpy(), scores)
    
    # 7. Walk-Forward Drift Validation
    print("\n--- Walk-Forward Validation ---")
    walk_forward_validation(dm, cfg, DEVICE, cls_w)
    
    # 8. Topological Manifold Forensics
    print("\n--- Manifold Visualization ---")
    visualize_manifold(dm, slice_t=42, emb_dim=3)

if __name__ == "__main__":
    main()
