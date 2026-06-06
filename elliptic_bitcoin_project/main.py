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
from evaluation.validation import walk_forward_validation, stack_prop, fit_head
from analysis.manifold_visualization import visualize_manifold
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore", category=UserWarning)

def plot_latent_space_kde(dm: EllipticDataModule, cfg: Config, slice_t: int = 42) -> None:
    g = dm.graphs[slice_t]
    X = g["x"]
    ei = g["edge_index"]
    X_prop = sgc_propagate(X, ei, cfg.sgc_k, cfg.use_multiscale_prop).numpy()
    y = g["y"].numpy()
    
    mask = (y != -1)
    X_known = X_prop[mask]
    y_known = y[mask]
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=cfg.seed)
    X_2d = reducer.fit_transform(X_known)
    
    licit_mask = (y_known == 0)
    illicit_mask = (y_known == 1)
    
    plt.figure(figsize=(12, 10), facecolor="#080812")
    ax = plt.gca()
    ax.set_facecolor="#080812")
    
    sns.kdeplot(x=X_2d[licit_mask, 0], y=X_2d[licit_mask, 1], cmap="Blues", fill=True, alpha=0.6, levels=10, thresh=0.05, ax=ax)
    sns.kdeplot(x=X_2d[illicit_mask, 0], y=X_2d[illicit_mask, 1], cmap="Reds", fill=True, alpha=0.6, levels=6, thresh=0.05, ax=ax)
    
    plt.scatter(X_2d[licit_mask, 0], X_2d[licit_mask, 1], s=2, c="#4C72B0", alpha=0.1, label="Licit Nodes")
    plt.scatter(X_2d[illicit_mask, 0], X_2d[illicit_mask, 1], s=15, c="#C44E52", alpha=0.9, edgecolors="white", linewidth=0.5, label="Illicit Nodes")
    
    plt.title(f"Latent Space Topography (SGC + UMAP) - t={slice_t}", color="#dddddd", fontsize=16, pad=20)
    plt.xlabel("UMAP 1", color="#aaaaaa"); plt.ylabel("UMAP 2", color="#aaaaaa")
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values(): spine.set_color('#333333')
    plt.legend(facecolor="#111111", edgecolor="#333333", labelcolor="#dddddd")
    
    import os
    from config import OUTPUT_DIR
    out_file = os.path.join(OUTPUT_DIR, f"latent_space_kde_t{slice_t}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="#080812")
    plt.close()

def run_defense_analytics(dm: EllipticDataModule, cfg: Config, model) -> None:
    print("\n--- Defense Analytics: SHAP & Complementary Error ---")
    Xs_f, ys_f, Xs_p, ys_p = [], [], [], []
    for t in cfg.train_steps:
        m = dm.graphs[t]["labeled_mask"].numpy()
        Xs_f.append(dm.graphs[t]["x"].numpy()[:, :166][m])
        ys_f.append(dm.graphs[t]["y"].numpy()[m])
    for t in cfg.test_steps:
        m = dm.graphs[t]["labeled_mask"].numpy()
        Xs_p.append(dm.graphs[t]["x"].numpy()[:, :166][m])
        ys_p.append(dm.graphs[t]["y"].numpy()[m])
        
    Xtr = np.concatenate(Xs_f); ytr = np.concatenate(ys_f)
    Xte = np.concatenate(Xs_p); yte = np.concatenate(ys_p)
    
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw, eval_metric="aucpr", random_state=cfg.seed, n_jobs=1).fit(Xtr, ytr)
    y_pred_xgb = (xgb.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    
    Xps, _ = stack_prop(dm, list(cfg.test_steps))
    with torch.no_grad():
        b2_all = torch.softmax(model(Xps.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        
    masks_p = [dm.graphs[t]["labeled_mask"].numpy() for t in cfg.test_steps]
    y_pred_sign = (b2_all[np.concatenate(masks_p)] >= 0.5).astype(int)
    
    illicit_mask = (yte == 1)
    xgb_fn = illicit_mask & (y_pred_xgb == 0)
    sign_fn = illicit_mask & (y_pred_sign == 0)
    
    print(f"Total Illicit Nodes in Test Set: {illicit_mask.sum()}")
    print(f"XGBoost Missed: {xgb_fn.sum()} | SIGN Missed: {sign_fn.sum()}")
    print(f"Illicit nodes missed by XGBoost but RECOVERED by SIGN: {(xgb_fn & (y_pred_sign == 1)).sum()}")
    
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(Xte)
    print(f"Total Absolute SHAP Importance (Local Ego Features 0-93): {np.abs(shap_values[:, :94]).sum():.2f}")
    print(f"Total Absolute SHAP Importance (Neighborhood Features 94-165): {np.abs(shap_values[:, 94:166]).sum():.2f}")

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
    walk_forward_validation(dm, cfg, DEVICE, sweep_name="main")
    
    # 8. Topological Manifold Forensics
    print("\n--- Manifold Visualization ---")
    visualize_manifold(dm, slice_t=42, emb_dim=3)
    
    print("\n--- Latent Space KDE Topography ---")
    plot_latent_space_kde(dm, cfg, slice_t=42)
    
    # 9. Final Analytics
    run_defense_analytics(dm, cfg, model)

if __name__ == "__main__":
    main()
