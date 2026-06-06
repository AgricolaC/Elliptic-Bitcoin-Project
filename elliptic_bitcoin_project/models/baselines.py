import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any, List
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from config import Config

def assemble_static(dm: Any, steps: Tuple[int, ...], raw_166_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Stack labeled nodes across `steps`. raw_166_only -> exclude engineered cols."""
    Xs, ys = [], []
    for t in steps:
        g = dm.graphs[t]
        m = g["labeled_mask"].numpy()
        # Ensure we only take the first NODE_FEATURE_DIM features for baselines
        x = g["x_np"][:, :len(dm.feature_cols)] if raw_166_only else g["x_np"]
        Xs.append(x[m])
        ys.append(g["y"].numpy()[m])
    return np.concatenate(Xs), np.concatenate(ys)

def report(name: str, y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_score >= thr).astype(int)
    out = dict(model=name,
               f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
               precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
               recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
               pr_auc=average_precision_score(y_true, y_score))
    print(f"  {name:16s} | illicit-F1={out['f1']:.3f}  P={out['precision']:.3f}  "
          f"R={out['recall']:.3f}  PR-AUC={out['pr_auc']:.3f}")
    return out

def run_baselines(dm: Any, cfg: Config) -> List[Dict[str, Any]]:
    Xtr_b, ytr_b = assemble_static(dm, cfg.train_steps)
    Xte_b, yte_b = assemble_static(dm, cfg.test_steps)
    
    print(f"Baseline train={Xtr_b.shape} test={Xte_b.shape} | "
          f"train illicit={ytr_b.mean():.3f} test illicit={yte_b.mean():.3f}")

    results = []
    
    # XGBoost Baseline
    spw = (ytr_b == 0).sum() / max((ytr_b == 1).sum(), 1)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=spw, eval_metric="aucpr", 
                        random_state=cfg.seed, n_jobs=1)
    xgb.fit(Xtr_b, ytr_b)
    results.append(report("XGBoost (166)", yte_b, xgb.predict_proba(Xte_b)[:, 1]))

    # RandomForest Baseline
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=1, random_state=cfg.seed)
    rf.fit(Xtr_b, ytr_b)
    results.append(report("RandomForest(166)", yte_b, rf.predict_proba(Xte_b)[:, 1]))
    
    return results
