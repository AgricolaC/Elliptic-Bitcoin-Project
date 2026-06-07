import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from typing import Any
import os
from config import Config, OUTPUT_DIR
from sklearn.metrics import f1_score, average_precision_score

def explicit_drift_adaptation(dm: Any, cfg: Config) -> dict:
    """Phase 9 - Explicit Drift Adaptation (surviving t=43)"""
    DRIFT_DECAY, DRIFT_WINDOW = 0.3, 8
    
    def xgb_walkforward(weight_fn=None, window=None):
        steps, f1s, praucs = [], [], []
        y_true_all, y_pred_all, y_score_all = [], [], []
        for tau in cfg.test_steps:
            tb = list(range(max(1, tau - window), tau)) if window else list(range(1, tau))
            Xs, ys, mts = [], [], []
            for t in tb:
                g = dm.graphs[t]
                m = g["labeled_mask"].numpy()
                Xs.append(g["x"].numpy()[:, :166][m])
                ys.append(g["y"].numpy()[m])
                mts.append(np.full(m.sum(), t))
            Xtr = np.concatenate(Xs)
            ytr = np.concatenate(ys)
            mt = np.concatenate(mts)
            
            w = weight_fn(tau, mt) if weight_fn else None
            
            g_te = dm.graphs[tau]
            m_te = g_te["labeled_mask"].numpy()
            Xte = g_te["x"].numpy()[:, :166][m_te]
            yte = g_te["y"].numpy()[m_te]
            
            if len(yte) == 0: continue
            
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=spw, eval_metric="aucpr",
                                random_state=cfg.seed, n_jobs=1).fit(Xtr, ytr, sample_weight=w)
            s = clf.predict_proba(Xte)[:, 1]
            y_pred = (s >= 0.5).astype(int)
            
            y_true_all.append(yte)
            y_pred_all.append(y_pred)
            y_score_all.append(s)
            
            steps.append(tau)

        from evaluation.validation import _aggregate_walk_forward
        pooled_f1, pooled_prauc, macro_f1, macro_prauc = _aggregate_walk_forward(y_true_all, y_pred_all, y_score_all)
        return steps, pooled_f1, pooled_prauc, macro_f1, macro_prauc

    st, pf1_u, pprauc_u, mf1_u, mprauc_u = xgb_walkforward()
    _,  pf1_r, pprauc_r, mf1_r, mprauc_r = xgb_walkforward(weight_fn=lambda tau, mt: np.exp(-DRIFT_DECAY * (tau - mt)))
    _,  pf1_s, pprauc_s, mf1_s, mprauc_s = xgb_walkforward(window=DRIFT_WINDOW)
    
    print(f"\n--- XGBoost Drift Adaptation Results ---")
    print(f"Uniform Expanding | Pooled F1={pf1_u:.3f} | Macro F1={mf1_u:.3f}")
    print(f"Recency-Weighted  | Pooled F1={pf1_r:.3f} | Macro F1={mf1_r:.3f}")
    print(f"Sliding (W={DRIFT_WINDOW})      | Pooled F1={pf1_s:.3f} | Macro F1={mf1_s:.3f}\n")
    
    # plt.figure(figsize=(11, 4.5))
    # plt.plot(st, f1_uniform, marker="o", label=f"uniform expanding (Pooled F1: {pf1_u:.3f})", color="#4C72B0")
    # plt.plot(st, f1_recency, marker="s", label=f"recency-weighted (Pooled F1: {pf1_r:.3f})", color="#C44E52")
    # plt.plot(st, f1_sliding, marker="^", label=f"sliding window W={DRIFT_WINDOW} (Pooled F1: {pf1_s:.3f})", color="#55A868")
    # plt.axvline(cfg.disruption_step, ls="--", color="k", label="t=43")
    # plt.xlabel("test step τ"); plt.ylabel("illicit F1"); plt.title("Drift adaptation strategies (Macro plot)")
    # plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    # out_file = os.path.join(OUTPUT_DIR, "drift_adaptation.png")
    # plt.savefig(out_file)
    # plt.close()
    
    return {
        "Sweep": "Phase 9: Drift Adaptation (Sliding Window)",
        "Static OOT F1": "N/A", 
        "Static OOT PR-AUC": "N/A",
        "Walk-Forward Mean F1": round(pf1_s, 3),
        "Walk-Forward Mean PR-AUC": round(pprauc_s, 3)
    }
