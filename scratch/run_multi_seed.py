import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Tuple

# Setup path
sys.path.append(os.path.join(os.getcwd(), "elliptic_bitcoin_project"))

from config import Config, set_global_seeds, DEVICE
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import stack_prop, _compute_class_weights
from models.classifier import SGCHead, build_loss
from models.layers import gcn_norm, _row_normalize

# Local patched sgc_propagate versions
def sgc_propagate_raw(x: torch.Tensor, edge_index: torch.Tensor, k: int, multiscale: bool) -> torch.Tensor:
    n, d = x.shape
    S = gcn_norm(edge_index, n)
    hops = [x]
    cur = x
    for _ in range(k):
        cur = torch.sparse.mm(S, cur)
        hops.append(cur)
    if multiscale:
        out = torch.cat(hops, dim=1)
    else:
        out = hops[-1]
    return out

def sgc_propagate_norm(x: torch.Tensor, edge_index: torch.Tensor, k: int, multiscale: bool) -> torch.Tensor:
    n, d = x.shape
    S = gcn_norm(edge_index, n)
    hops = [x]
    cur = x
    for _ in range(k):
        cur = torch.sparse.mm(S, cur)
        hops.append(cur)
    if multiscale:
        normalized = [hops[0]]
        for h in hops[1:]:
            std = h.std(dim=0, keepdim=True).clamp(min=1e-6)
            normalized.append(h / std)
        out = torch.cat(normalized, dim=1)
    else:
        out = hops[-1]
    return out

def custom_fit_head(Xtr, ytr, in_dim, cfg, class_weights, device, use_adamw=False):
    set_global_seeds(cfg.seed)
    model = SGCHead(in_dim, cfg).to(device)
    loss_fn = build_loss(cfg, class_weights)
    if use_adamw:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay)
    
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    mask = (ytr != -1)
    model.train()
    for _ in range(cfg.sgc_epochs):
        opt.zero_grad()
        if mask.sum() > 0:
            loss = loss_fn(model(Xtr)[mask], ytr[mask])
            loss.backward()
            opt.step()
    return model

def evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale, norm_mode, use_adamw):
    from sklearn.metrics import f1_score
    f1s = []
    cfg = Config(use_mlp_head=True, use_multiscale_prop=use_multiscale, use_graph_structural=False)
    
    # We patch sgc_propagate dynamically in a dummy class to pass into dm
    for seed in seeds:
        cfg.seed = seed
        set_global_seeds(seed)
        
        # Patch the dm's sgc_propagate
        import data.build_graph as bg
        if norm_mode == "raw":
            bg.sgc_propagate = lambda x, ei, k, ms, ud: sgc_propagate_raw(x, ei, k, ms)
        elif norm_mode == "norm":
            bg.sgc_propagate = lambda x, ei, k, ms, ud: sgc_propagate_norm(x, ei, k, ms)
            
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()
        
        Xtr, ytr = stack_prop(dm, list(cfg.train_steps))
        Xte, yte = stack_prop(dm, list(cfg.test_steps))
        
        valid_ytr = ytr[ytr != -1]
        counts = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
        cls_w = (counts.sum() / (2.0 * counts)).to(DEVICE)
        
        model = custom_fit_head(Xtr, ytr, dm.sgc_input_dim, cfg, cls_w, DEVICE, use_adamw)
        model.eval()
        with torch.no_grad():
            m = (yte != -1)
            scores = torch.softmax(model(Xte[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        
        y_true = yte[m].numpy()
        f1 = f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)
        f1s.append(f1)
        
    return np.mean(f1s), np.std(f1s)

def main():
    print("Loading data...")
    df, df_edge, _, feature_cols = download_and_load_data()
    seeds = [42, 43, 44, 45, 46]
    
    print("\n--- Running Adam (Coupled Weight Decay) ---")
    mean, std = evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale=False, norm_mode="raw", use_adamw=False)
    print(f"Sweep 2 (S^2 X, in=166)      : {mean:.3f} ± {std:.3f}")
    
    mean, std = evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale=True, norm_mode="raw", use_adamw=False)
    print(f"Sweep 3 Raw (in=498)         : {mean:.3f} ± {std:.3f}")
    
    mean, std = evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale=True, norm_mode="norm", use_adamw=False)
    print(f"Sweep 3 Norm (in=498)        : {mean:.3f} ± {std:.3f}")

    print("\n--- Running AdamW (Decoupled Weight Decay) ---")
    mean, std = evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale=False, norm_mode="raw", use_adamw=True)
    print(f"Sweep 2 (S^2 X, in=166)      : {mean:.3f} ± {std:.3f}")
    
    mean, std = evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale=True, norm_mode="raw", use_adamw=True)
    print(f"Sweep 3 Raw (in=498)         : {mean:.3f} ± {std:.3f}")
    
    mean, std = evaluate_config(df, df_edge, feature_cols, seeds, use_multiscale=True, norm_mode="norm", use_adamw=True)
    print(f"Sweep 3 Norm (in=498)        : {mean:.3f} ± {std:.3f}")

if __name__ == "__main__":
    main()
