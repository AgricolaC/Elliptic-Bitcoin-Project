import numpy as np
from xgboost import XGBClassifier
from typing import Any
from config import Config
from evaluation.validation import stack_prop

def pu_learning_adjust(dm: Any, cfg: Config) -> dict:
    """Phase 8 - Exploit the 77% unlabeled nodes using Positive-Unlabeled Learning (Elkan-Noto)."""
    rng_pu = np.random.default_rng(cfg.seed)
    
    # Assemble raw base features across all training nodes
    Xs, ys = [], []
    for t in cfg.train_steps:
        g = dm.graphs[t]
        Xs.append(g["x"].numpy())
        ys.append(g["y"].numpy())
    Xtr = np.concatenate(Xs)
    ytr = np.concatenate(ys)
    
    mask_p = (ytr == 1)
    mask_u = (ytr == -1)
    
    Xp = Xtr[mask_p]
    Xu = Xtr[mask_u]
    
    if len(Xu) > 5 * len(Xp):                              # cap U for balance/speed
        Xu = Xu[rng_pu.choice(len(Xu), 5 * len(Xp), replace=False)]

    # LEAKAGE GUARD: hold out a slice of positives for the c estimate (not used in fit)
    perm = rng_pu.permutation(len(Xp))
    val = perm[:len(Xp)//5]
    fit_p = perm[len(Xp)//5:]
    
    X_pu = np.vstack([Xp[fit_p], Xu])
    s_pu = np.r_[np.ones(len(fit_p)), np.zeros(len(Xu))]
    
    clf_pu = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=len(Xu)/max(len(fit_p),1), eval_metric="aucpr",
                           random_state=cfg.seed, n_jobs=1).fit(X_pu, s_pu)
                           
    c = float(clf_pu.predict_proba(Xp[val])[:, 1].mean())  # label propensity
    print(f"PU Learning estimated label propensity c = {c:.3f}")
    
    # Evaluate on test set
    Xte_s, yte_s = [], []
    for t in cfg.test_steps:
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        Xte_s.append(g["x"].numpy()[m])
        yte_s.append(g["y"].numpy()[m])
    Xte_labeled = np.concatenate(Xte_s)
    yte_labeled = np.concatenate(yte_s)
    
    pu_score = np.clip(clf_pu.predict_proba(Xte_labeled)[:, 1] / max(c, 1e-6), 0, 1)
    
    from sklearn.metrics import f1_score, average_precision_score
    y_pred = (pu_score >= 0.5).astype(int)
    f1 = f1_score(yte_labeled, y_pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(yte_labeled, pu_score)
    
    return {"Sweep": "Phase 8: PU(illicit/unknown)", "Static OOT F1": round(f1, 3), "Static OOT PR-AUC": round(pr_auc, 3), "Walk-Forward Mean F1": "N/A", "Walk-Forward Mean PR-AUC": "N/A"}
