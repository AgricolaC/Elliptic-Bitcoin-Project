import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Any
from sklearn.metrics import f1_score, average_precision_score
from config import Config, OUTPUT_DIR, set_global_seeds
from models.classifier import SGCHead, build_loss

def stack_prop(dm: Any, steps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ALL nodes' propagated features over `steps`."""
    Xs, ys = [], []
    for t in steps:
        g = dm.graphs[t]
        Xs.append(g["prop"])
        ys.append(g["y"])
    return torch.cat(Xs), torch.cat(ys)

def fit_head(Xtr: torch.Tensor, ytr: torch.Tensor, in_dim: int, cfg: Config, class_weights: torch.Tensor, device: torch.device) -> SGCHead:
    """Train one SGCHead for Walk-Forward."""
    set_global_seeds(cfg.seed)
    model = SGCHead(in_dim, cfg).to(device)
    loss_fn = build_loss(cfg, class_weights)
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

def walk_forward_validation(dm: Any, cfg: Config, device: torch.device, cls_w: torch.Tensor) -> Tuple[float, float]:
    wf_steps, wf_f1, wf_prauc = [], [], []
    
    for tau in cfg.test_steps:
        train_block = list(range(1, tau))
        assert max(train_block) == tau - 1  # LEAKAGE GUARD
        
        Xtr_w, ytr_w = stack_prop(dm, train_block)
        
        g = dm.graphs[tau]
        m = g["labeled_mask"]
        Xte_w = g["prop"][m]
        yte_w = g["y"][m].numpy()
        
        if m.sum() == 0 or len(np.unique(yte_w)) < 2:
            continue
            
        model = fit_head(Xtr_w, ytr_w, dm.sgc_input_dim, cfg, cls_w, device)
        model.eval()
        
        with torch.no_grad():
            s = torch.softmax(model(Xte_w.to(device)), dim=1)[:, 1].cpu().numpy()
            
        wf_steps.append(tau)
        wf_f1.append(f1_score(yte_w, (s >= 0.5).astype(int), pos_label=1, zero_division=0))
        wf_prauc.append(average_precision_score(yte_w, s))

    plt.figure(figsize=(11, 4.5))
    plt.plot(wf_steps, wf_f1, marker="o", label="Illicit F1", color="#C44E52")
    plt.plot(wf_steps, wf_prauc, marker="s", label="PR-AUC", color="#4C72B0")
    plt.axvline(cfg.disruption_step, ls="--", color="k", label="t=43 dark-market shutdown")
    plt.xlabel("Test time step τ")
    plt.ylabel("score")
    plt.title(f"Walk-forward (multiscale={cfg.use_multiscale_prop}, mlp={cfg.use_mlp_head}, focal={cfg.use_focal_loss})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "walk_forward_drift.png")
    plt.savefig(out_file)
    plt.close()
    
    mean_f1 = np.mean(wf_f1)
    mean_prauc = np.mean(wf_prauc)
    print(f"Mean walk-forward F1={mean_f1:.3f} | min F1={min(wf_f1):.3f} at τ={wf_steps[int(np.argmin(wf_f1))]}")
    return float(mean_f1), float(mean_prauc)
