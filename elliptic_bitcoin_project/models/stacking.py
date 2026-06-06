import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from typing import Any, List, Tuple
from config import Config, DEVICE

try:
    from evaluation.validation import fit_head, stack_prop
except ImportError:
    fit_head = stack_prop = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def _base_predictions(
    fit_steps: List[int],
    pred_steps: List[int],
    dm: Any,
    cfg: Config,
) -> Tuple:
    """
    Generate base model predictions for the stacking ensemble.

    W2 FIX: This function is now the single place where base models are fitted.
    Both the meta-training call (early -> late) and the meta-test call
    (early -> test) pass the SAME fit_steps (early window only).
    This eliminates the covariate shift that occurred when the test call
    previously used cfg.train_steps as fit_steps.

    Args:
        fit_steps:  Timesteps used to TRAIN the base models.
        pred_steps: Timesteps for which to GENERATE predictions.
        dm:         EllipticDataModule with populated graphs.
        cfg:        Configuration object.

    Returns:
        Tuple of (Z: np.ndarray [n_pred, 3], y: np.ndarray [n_pred])
    """
    # SHAPE GUARD
    assert len(fit_steps) > 0,  "fit_steps must be non-empty."
    assert len(pred_steps) > 0, "pred_steps must be non-empty."
    assert set(fit_steps).isdisjoint(set(pred_steps)), \
        "LEAKAGE GUARD: fit_steps and pred_steps must be disjoint."

    Xs_f, ys_f = [], []
    Xs_p, ys_p = [], []

    for t in fit_steps:
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        Xs_f.append(g["x"].numpy()[:, :166][m])
        ys_f.append(g["y"].numpy()[m])

    for t in pred_steps:
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        Xs_p.append(g["x"].numpy()[:, :166][m])
        ys_p.append(g["y"].numpy()[m])

    Xf  = np.concatenate(Xs_f)
    yf  = np.concatenate(ys_f)
    Xp  = np.concatenate(Xs_p)
    yp  = np.concatenate(ys_p)
    spw = (yf == 0).sum() / max((yf == 1).sum(), 1)

    b1 = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=cfg.seed, n_jobs=1,
    ).fit(Xf, yf).predict_proba(Xp)[:, 1]

    # TDA-proxy view: full augmented features
    Xs_ft, Xs_pt = [], []
    for t in fit_steps:
        g = dm.graphs[t]; m = g["labeled_mask"].numpy()
        Xs_ft.append(g["x"].numpy()[m])
    for t in pred_steps:
        g = dm.graphs[t]; m = g["labeled_mask"].numpy()
        Xs_pt.append(g["x"].numpy()[m])

    Xft = np.concatenate(Xs_ft)
    Xpt = np.concatenate(Xs_pt)
    b3  = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=cfg.seed, n_jobs=1,
    ).fit(Xft, yf).predict_proba(Xpt)[:, 1]

    # Neural SIGN view
    Xfs, yfs = stack_prop(dm, list(fit_steps))
    Xps, _   = stack_prop(dm, list(pred_steps))
    counts   = torch.bincount(yfs[yfs != -1], minlength=2).float()
    cls_w    = (counts.sum() / (2 * counts)).to(DEVICE)
    sign     = fit_head(Xfs, yfs, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    sign.eval()
    with torch.no_grad():
        b2 = torch.softmax(sign(Xps.to(DEVICE)), dim=1)[:, 1].cpu().numpy()

    # SHAPE GUARD: all three base predictions must align
    assert b1.shape == b2.shape == b3.shape == yp.shape, (
        f"Base prediction shape mismatch: b1={b1.shape}, b2={b2.shape}, "
        f"b3={b3.shape}, yp={yp.shape}"
    )
    return np.column_stack([b1, b2, b3]), yp


def stacking_meta_classifier(dm: Any, cfg: Config) -> dict:
    """
    Phase 11 — Stacking (tree ⊕ SIGN ⊕ TDA).

    W2 FIX: Both the meta-training call and the meta-test call use `early`
    as fit_steps. The meta-LR is therefore trained on predictions from models
    fitted on early steps, and evaluated on predictions from models ALSO fitted
    on early steps only. This eliminates the covariate distribution shift
    that occurred when the test call used cfg.train_steps.

    Temporal ordering guarantee:
        early  → predicts  late   → meta-LR trained
        early  → predicts  test   → meta-LR evaluated
    """
    mid = (max(cfg.train_steps) + min(cfg.train_steps)) // 2
    early = [s for s in cfg.train_steps if s <= mid]
    late  = [s for s in cfg.train_steps if s > mid]

    # LEAKAGE GUARD
    assert set(early).isdisjoint(set(late)), \
        "early and late windows must be disjoint."
    assert max(early) < min(late), \
        "early window must strictly precede late window (temporal order)."
    assert max(late) < min(cfg.test_steps), \
        "late window must strictly precede test steps."

    # Meta-train: base models fitted on early, predict on late
    Z_late, y_late = _base_predictions(early, late, dm, cfg)
    meta = LogisticRegression(class_weight="balanced", max_iter=500).fit(Z_late, y_late)

    # Meta-test: base models fitted on early (SAME window — W2 fix), predict on test
    Z_test, y_test = _base_predictions(early, list(cfg.test_steps), dm, cfg)
    stack_score    = meta.predict_proba(Z_test)[:, 1]

    f1     = f1_score(y_test, (stack_score >= 0.5).astype(int), pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_test, stack_score)

    return {
        "Sweep":                  "Phase 11: STACK(tree⊕SIGN⊕TDA)",
        "Static OOT F1":          round(f1, 3),
        "Static OOT PR-AUC":      round(pr_auc, 3),
        "Walk-Forward Mean F1":   "N/A",
        "Walk-Forward Mean PR-AUC": "N/A",
    }


# Re-export for backward compat
