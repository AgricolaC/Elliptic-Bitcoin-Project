import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from typing import Any
from config import Config, DEVICE
from evaluation.validation import fit_head, stack_prop

def stacking_meta_classifier(dm: Any, cfg: Config) -> dict:
    """Phase 11 - Stacking the Three Views (tree ⊕ SIGN ⊕ TDA)"""
    
    early = [s for s in cfg.train_steps if s <= 24]
    late  = [s for s in cfg.train_steps if 24 < s <= 34]
    assert set(early).isdisjoint(late)
    
    def base_predictions(fit_steps, pred_steps):
        # Tree 1: Raw 166 features
        Xs_f, ys_f = [], []
        Xs_p, ys_p = [], []
        for t in fit_steps:
            g = dm.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_f.append(g["x"].numpy()[:, :166][m])
            ys_f.append(g["y"].numpy()[m])
        for t in pred_steps:
            g = dm.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_p.append(g["x"].numpy()[:, :166][m])
            ys_p.append(g["y"].numpy()[m])
            
        Xf, yf = np.concatenate(Xs_f), np.concatenate(ys_f)
        Xp, yp = np.concatenate(Xs_p), np.concatenate(ys_p)
        
        spw = (yf == 0).sum() / max((yf == 1).sum(), 1)
        b1 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
                           eval_metric="aucpr", random_state=cfg.seed, n_jobs=1).fit(Xf, yf).predict_proba(Xp)[:, 1]
                           
        # Tree 2: Topology features (simulating TDA view using base+topo features since Ripser is excluded)
        # Note: cfg.use_topology must be True
        Xs_ft, Xs_pt = [], []
        for t in fit_steps:
            g = dm.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_ft.append(g["x"].numpy()[m])
        for t in pred_steps:
            g = dm.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_pt.append(g["x"].numpy()[m])
            
        Xft, Xpt = np.concatenate(Xs_ft), np.concatenate(Xs_pt)
        b3 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
                           eval_metric="aucpr", random_state=cfg.seed, n_jobs=1).fit(Xft, yf).predict_proba(Xpt)[:, 1]
                           
        # Neural View: SIGN
        Xfs, yfs = stack_prop(dm, fit_steps)
        Xps, _ = stack_prop(dm, pred_steps)
        
        counts = torch.bincount(yfs[yfs != -1], minlength=2).float()
        cls_w = (counts.sum() / (2 * counts)).to(DEVICE)
        
        sign = fit_head(Xfs, yfs, dm.sgc_input_dim, cfg, cls_w, DEVICE)
        sign.eval()
        with torch.no_grad():
            b2 = torch.softmax(sign(Xps.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            
        return np.column_stack([b1, b2, b3]), yp
        
    Z_late, y_late = base_predictions(early, late)
    meta = LogisticRegression(class_weight="balanced", max_iter=500).fit(Z_late, y_late)
    
    Z_test, y_test = base_predictions(cfg.train_steps, cfg.test_steps)
    stack_score = meta.predict_proba(Z_test)[:, 1]
    
    y_pred = (stack_score >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_test, stack_score)
    
    # We won't simulate walk_forward for stacking to save computation time
    
    return {"Sweep": "Phase 11: STACK(tree⊕SIGN⊕TDA)", "Static OOT F1": round(f1, 3), "Static OOT PR-AUC": round(pr_auc, 3), "Walk-Forward Mean F1": "N/A", "Walk-Forward Mean PR-AUC": "N/A"}
