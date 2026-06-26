import numpy as np
import torch
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Any
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from config import Config, OUTPUT_DIR, set_global_seeds

try:
    from models.classifier import SGCHead, build_loss
except ImportError:
    SGCHead = build_loss = None

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k scored items that are anomalous."""
    top_k = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k].sum()) / k


def _find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold maximizing illicit-class F1.

    Uses [:-1] slicing because precision_recall_curve returns
    precisions/recalls of length n+1 but thresholds of length n.
    Without this, argmax can land on the final element and index
    out of bounds — a realistic crash on degenerate steps.

    Returns 0.5 as fallback if no valid threshold can be found.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    if len(f1s) == 0 or f1s.max() == 0:
        return 0.5  # fallback
    return float(thresholds[np.argmax(f1s)])


def _calibrate_threshold(y_cal: np.ndarray, s_cal: np.ndarray,
                         global_illicit_rate: float = None,
                         epsilon: int = 10) -> Tuple[float, bool]:
    """Pick the operating threshold from the calibration step.

    Under prevalence collapse (the τ=43 regime shift) the calibration step can
    have almost no positives, making the supervised F1-threshold pure noise
    (World C). When the calibration step has fewer than ``epsilon`` positives,
    fall back to a LOCAL quantile: the ``1 - local_rate`` quantile of the
    calibration score distribution, where ``local_rate`` is the observed
    illicit fraction in the calibration step itself.  This honours the
    post-break empirical prevalence rather than imposing a pre-break global
    base-rate onto a structurally different regime.

    Returns ``(threshold, fallback_fired)``.
    """
    n_pos = int((y_cal == 1).sum())
    if n_pos < epsilon:
        rate = global_illicit_rate if global_illicit_rate is not None else (n_pos / max(len(y_cal), 1))
        q = float(np.quantile(s_cal, 1.0 - rate))
        return q, True
    return float(_find_best_f1_threshold(y_cal, s_cal)), False


def _aggregate_walk_forward(y_true_list: List[np.ndarray], y_pred_list: List[np.ndarray], score_list: List[np.ndarray]) -> Tuple[float, float, float, float, float]:
    """
    Compute pooled and macro-averaged walk-forward metrics.
    Returns: (pooled_f1, pooled_prauc, macro_f1, macro_prauc)
    """
    yt_p = np.concatenate(y_true_list)
    yp_p = np.concatenate(y_pred_list)
    ys_p = np.concatenate(score_list)
    
    pooled_f1 = float(f1_score(yt_p, yp_p, pos_label=1, zero_division=0))
    pooled_prauc = float(average_precision_score(yt_p, ys_p))
    k = int(yt_p.sum())
    pooled_pak = precision_at_k(yt_p, ys_p, k) if k > 0 else 0.0
    
    f1s, praucs = [], []
    for yt, yp, ys in zip(y_true_list, y_pred_list, score_list):
        if len(np.unique(yt)) >= 2:
            f1s.append(float(f1_score(yt, yp, pos_label=1, zero_division=0)))
            praucs.append(float(average_precision_score(yt, ys)))
        else:
            f1s.append(0.0)
            praucs.append(0.0)
            
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_prauc = float(np.mean(praucs)) if praucs else 0.0
    
    return pooled_f1, pooled_prauc, macro_f1, macro_prauc, pooled_pak


def stack_prop(dm: Any, steps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ALL nodes' propagated features over `steps`."""
    expected_dim = dm.sgc_input_dim
    Xs, ys = [], []
    for t in steps:
        assert t in dm.graphs, f"step {t} not in dm.graphs"
        g = dm.graphs[t]
        assert g["prop"].shape[1] == expected_dim, \
            f"step {t} prop width {g['prop'].shape[1]} != sgc_input_dim {expected_dim}"
        Xs.append(g["prop"])
        ys.append(g["y"])
    return torch.cat(Xs), torch.cat(ys)


def fit_head(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    in_dim: int,
    cfg: Config,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = -1,
) -> "SGCHead":
    """
    Train one SGCHead.

    Defensive Notes:
        - Seeds reset before every fit call for reproducibility.
        - Labeled mask applied inside loop: unknown nodes (-1) excluded from loss.
    """
    set_global_seeds(cfg.seed)
    model = SGCHead(in_dim, cfg).to(device)
    loss_fn = build_loss(cfg, class_weights)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay
    )
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    mask = (ytr != -1)

    model.train()
    n_epochs = epochs if epochs > 0 else cfg.sgc_epochs
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        if mask.sum() > 0:
            loss = loss_fn(model(Xtr)[mask], ytr[mask])
            
            # L1 on first layer for feature selection
            first_layer = model.net[0]
            if hasattr(first_layer, 'weight'):
                l1_penalty = first_layer.weight.abs().sum()
            elif hasattr(first_layer, 'lin'):
                l1_penalty = first_layer.lin.weight.abs().sum()
            else:
                l1_penalty = 0.0
                
            if getattr(cfg, 'sgc_l1_lambda', 0.0) > 0.0:
                loss += cfg.sgc_l1_lambda * l1_penalty
            
            loss.backward()
            opt.step()
    return model


def _compute_class_weights(ytr: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from labeled nodes in ytr.

    Defensive Notes:
        - Only labeled nodes (y != -1) are used for counting.
        - Returns uniform weights if no labeled nodes exist.
    """
    labeled = ytr[ytr != -1]
    if labeled.numel() == 0:
        return torch.ones(2, device=device)
    counts = torch.bincount(labeled, minlength=2).float()
    # Guard against a class being absent in this window
    counts = counts.clamp(min=1.0)
    return (counts.sum() / (2.0 * counts)).to(device)



def walk_forward_validation(
    dm: Any,
    cfg: Config,
    device: torch.device,
    sweep_name: str = "default",   # W7 FIX: parameterized filename
    return_records: bool = False,
    window: int = None,
    eval_steps: range = None,       # P0-A: pass cfg.val_steps or cfg.test_steps
    xgb_dm: Any = None,             # Ensemble Addition
    xgb_weight: float = 0.5         # Ensemble Addition
) -> Any:
    """
    Walk-forward temporal validation with dynamic class weights.

    P1-A: Threshold is calibrated on a held-out step (τ-1), not in-sample.
    For each τ: train on [start..τ-2], calibrate threshold on τ-1, test on τ.
    Falls back to 0.5 when τ-1 has insufficient labeled data.

    NOTE: Preprocessing (scalers, topology, propagation) is frozen from setup().
    Only the classification head is retrained per-tau. This is a stated
    simplification, not leakage.

    W7 FIX: Output filename embeds sweep_name, preventing silent overwrite when
            multiple sweeps are run sequentially.

    W8 FIX: Class weights are recomputed inside the tau loop from the expanding
            training window [1..tau-2], not from a fixed static training split.

    Defensive Notes:
        - LEAKAGE GUARD: train_block = [start, tau-2]; calibration on tau-1.
        - Skips tau steps with no labeled nodes or single-class labels.
        - Class weights are computed before each model fit from actual window data.

    Args:
        dm:             EllipticDataModule with 'prop' keys set (by setup()).
        cfg:            Configuration object.
        device:         Torch device for model training.
        sweep_name:     Label embedded in the output filename.
        return_records: If True, return per-step records as third element.
        window:         If set, use a sliding window of this width. None = expanding.
        eval_steps:     Steps to evaluate on. Defaults to cfg.test_steps.
                        Pass cfg.val_steps for model selection.

    Returns:
        If return_records:  (pooled_f1, pooled_prauc, wf_records)
        Else:               (pooled_f1, pooled_prauc)
    """
    # Sanitize sweep_name for safe filesystem use
    safe_name = re.sub(r"[^\w\-]", "_", sweep_name)

    if eval_steps is None:
        eval_steps = cfg.test_steps

    wf_steps = []
    wf_f1_per_step = []
    wf_prauc_per_step = []
    wf_metadata = []
    wf_records = []
    y_true_all, y_pred_all, s_pred_all = [], [], []
    total_wf_train_time = 0.0

    for tau in eval_steps:
        start_t = max(min(dm.graphs), tau - window) if window else min(dm.graphs)
        # P1-A: train on [start..tau-2], calibrate threshold on tau-1
        train_block = list(range(start_t, tau - 1))
        train_block = [t for t in train_block if t in dm.graphs]
        calib_step = tau - 1

        if not train_block:
            continue

        # LEAKAGE GUARD: calib step must not appear in the training window.
        assert calib_step not in train_block, \
            f"LEAKAGE: calib_step={calib_step} found in train_block={train_block}"

        Xtr_w, ytr_w = stack_prop(dm, train_block)

        g     = dm.graphs[tau]
        m     = g["labeled_mask"]
        Xte_w = g["prop"][m]
        assert Xte_w.shape[1] == dm.sgc_input_dim, \
            f"Test prop width {Xte_w.shape[1]} != sgc_input_dim {dm.sgc_input_dim}"
        yte_w = g["y"][m].numpy()

        if m.sum() == 0:
            continue

        # Single-class skip guard: can't compute F1 with only one class
        if len(np.unique(yte_w)) < 2:
            continue

        # W8 FIX: compute weights from the expanding window, not static train_steps
        cls_w = _compute_class_weights(ytr_w, device)
        n_epochs = getattr(cfg, "wf_epochs", cfg.sgc_epochs)

        t0 = time.perf_counter()
        model = fit_head(Xtr_w, ytr_w, dm.sgc_input_dim, cfg, cls_w, device, epochs=n_epochs)
        
        model_xgb = None
        if xgb_dm is not None:
            from evaluation.ablation_validation import _tab_block
            from xgboost import XGBClassifier
            Xtr_xgb, ytr_xgb = _tab_block(xgb_dm, train_block)
            spw = (ytr_xgb == 0).sum() / max((ytr_xgb == 1).sum(), 1)
            model_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      scale_pos_weight=spw, eval_metric="aucpr", 
                                      random_state=getattr(cfg, "seed", 42), n_jobs=1)
            model_xgb.fit(Xtr_xgb, ytr_xgb)

        total_wf_train_time += (time.perf_counter() - t0)
        
        model.eval()

        # P1-A: Calibrate threshold on held-out step tau-1 (not in-sample).
        # Uses _calibrate_threshold (not _find_best_f1_threshold directly) so the
        # ε-fallback fires consistently across SGC and LSTM/EMA paths when the
        # calibration step has < ε positives (e.g., tau=44, calib=43=shock).
        # NOTE: when n_pos >= epsilon, _calibrate_threshold delegates to
        # _find_best_f1_threshold — so pre-shock F1 is identical across paths.
        threshold = 0.5  # fallback
        fallback = True
        if calib_step in dm.graphs:
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        s_cal = torch.softmax(
                            model(g_cal["prop"][m_cal].to(device)), dim=1
                        )[:, 1].cpu().numpy()
                    
                    if xgb_dm is not None and model_xgb is not None:
                        from evaluation.ablation_validation import _tab_step
                        Xc_xgb, yc_xgb = _tab_step(xgb_dm, calib_step)
                        s_cal_xgb = model_xgb.predict_proba(Xc_xgb)[:, 1]
                        s_cal = xgb_weight * s_cal_xgb + (1 - xgb_weight) * s_cal
                        
                    threshold, fallback = _calibrate_threshold(y_cal, s_cal)

        with torch.no_grad():
            # PRAUC COLUMN GUARD: [:, 1] is the illicit-class probability.
            # class_map: "1"→1 (illicit/positive), "2"→0 (licit/negative).
            # average_precision_score expects scores for the positive class.
            logits_te = model(Xte_w.to(device))
            assert logits_te.shape[1] == 2, \
                f"tau={tau}: expected 2-class logits, got shape {logits_te.shape}"
            s = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()
            
            if xgb_dm is not None and model_xgb is not None:
                from evaluation.ablation_validation import _tab_step
                Xte_xgb, yte_xgb = _tab_step(xgb_dm, tau)
                s_te_xgb = model_xgb.predict_proba(Xte_xgb)[:, 1]
                s = xgb_weight * s_te_xgb + (1 - xgb_weight) * s

        y_pred = (s >= threshold).astype(int)
        step_f1 = float(f1_score(yte_w, y_pred, pos_label=1, zero_division=0))
        step_prauc = float(average_precision_score(yte_w, s))

        wf_steps.append(tau)
        wf_f1_per_step.append(step_f1)
        wf_prauc_per_step.append(step_prauc)
        wf_metadata.append({
            "Train_Window_Size": len(train_block),
            "Calib_Threshold": round(float(threshold), 4),
            "Calib_Fallback": fallback
        })

        y_true_all.append(yte_w)
        s_pred_all.append(s)
        y_pred_all.append(y_pred)
        
        if return_records:
            wf_records.append((tau, step_f1, step_prauc, yte_w, s, y_pred))

    if wf_steps:
        plt.figure(figsize=(12, 5), facecolor="white")
        
        plt.plot(wf_steps, wf_f1_per_step,    marker="o", linewidth=2, label="Illicit F1",  color="#C44E52")
        plt.plot(wf_steps, wf_prauc_per_step, marker="s", linewidth=2, label="PR-AUC",      color="#4C72B0")
        
        plt.axvline(cfg.disruption_step, ls="--", color="black", linewidth=2,
                    label=f"dark market shutdown (t={cfg.disruption_step})")
        
        plt.xlabel("Test time step τ", fontsize=14, fontweight="bold")
        plt.ylabel("Score", fontsize=14, fontweight="bold")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.title(
            f"Walk-forward [{safe_name}] | "
            f"multiscale={cfg.use_multiscale_prop} | "
            f"struct={cfg.use_graph_structural}",
            fontsize=16, fontweight="bold"
        )
        
        plt.legend(fontsize=12, loc="upper right")
        plt.grid(alpha=0.4)
        plt.tight_layout()

        # W7 FIX: unique filename per sweep — no overwrites
        out_file = os.path.join(OUTPUT_DIR, f"walk_forward_drift_{safe_name}.png")
        plt.savefig(out_file)

        # Export timestep data to master CSV (16-column unified schema)
        import pandas as pd
        from evaluation.wf_metrics import regime_of
        from sklearn.metrics import precision_score, recall_score
        
        csv_file = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
        
        out_rows = []
        for i, step_tau in enumerate(wf_steps):
            step_f1 = wf_f1_per_step[i]
            step_prauc = wf_prauc_per_step[i]
            step_y_true = y_true_all[i]
            step_scores = s_pred_all[i]
            step_y_pred = y_pred_all[i]
            meta = wf_metadata[i]
            
            n_ill = int((step_y_true == 1).sum())
            n_lic = int((step_y_true == 0).sum())
            prec = float(precision_score(step_y_true, step_y_pred, pos_label=1, zero_division=0))
            rec = float(recall_score(step_y_true, step_y_pred, pos_label=1, zero_division=0))
            
            out_rows.append({
                "Sweep": sweep_name,
                "Seed": getattr(cfg, "seed", 42),
                "Tau": step_tau,
                "N_labeled": n_ill + n_lic,
                "N_illicit": n_ill,
                "N_licit": n_lic,
                "Low_Confidence": n_ill < 10,
                "Regime": regime_of(step_tau),
                "Train_Window_Size": meta["Train_Window_Size"],
                "Calib_Threshold": meta["Calib_Threshold"],
                "Calib_Fallback": meta["Calib_Fallback"],
                "F1": round(step_f1, 4),
                "PRAUC": round(step_prauc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "Selfcond_Bug": "fixed"
            })
            
        CSV2_COLS = ["Sweep", "Seed", "Tau", "N_labeled", "N_illicit", "N_licit",
                     "Low_Confidence", "Regime", "Train_Window_Size", "Calib_Threshold",
                     "Calib_Fallback", "F1", "PRAUC", "Precision", "Recall", "Selfcond_Bug"]
        df_new = pd.DataFrame(out_rows, columns=CSV2_COLS)
        
        if os.path.exists(csv_file):
            try:
                df_old = pd.read_csv(csv_file, keep_default_na=False)
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df_out = df_new
        else:
            df_out = df_new
        df_out.to_csv(csv_file, index=False)
        plt.close()

    if not wf_steps:
        return (0.0, 0.0, []) if return_records else (0.0, 0.0)

    pooled_f1, pooled_prauc, macro_f1, macro_prauc, pooled_pak = _aggregate_walk_forward(y_true_all, y_pred_all, s_pred_all)
    
    print(
        f"[{safe_name}] Pooled F1={pooled_f1:.3f} | Macro F1={macro_f1:.3f} | "
        f"Pooled PRAUC={pooled_prauc:.3f} | Macro PRAUC={macro_prauc:.3f} | "
        f"Pooled P@K={pooled_pak:.3f}"
    )
    print(f"  [WF Pipeline] Walk-forward explicit training loop overhead: {total_wf_train_time:.2f} seconds")
    return (pooled_f1, pooled_prauc, wf_records) if return_records else (pooled_f1, pooled_prauc)
