"""
F4 Exponential Decay Ablation Study
Tests Walk-Forward Validation with exponentially decayed sample weighting
for both XGBoost and SGC+MLP.

Results are written to:
  - results/sweep_results.csv      (one aggregate row per model × lambda)
  - results/walk_forward_timesteps.csv  (one per-τ row per model × lambda)
"""
import sys
import os
import time
import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, average_precision_score

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.build_graph import EllipticDataModule
from execution.phases.f1_walk_forward import _tab_block, _tab_step
from evaluation.validation import SGCHead, set_global_seeds
from evaluation.wf_metrics import stratified_wf_metrics
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")
TS_CSV    = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
SEED      = 42

# ─── CSV-2 schema (matches f1_walk_forward.py) ────────────────────────────────
CSV2_COLS = ["Sweep", "Seed", "Tau", "N_labeled", "N_illicit", "N_licit",
             "Low_Confidence", "Regime", "Train_Window_Size", "Calib_Threshold",
             "Calib_Fallback", "F1", "PRAUC", "Precision", "Recall", "Selfcond_Bug"]


def _write_csv2(sweep, rows, train_window_sizes, thresholds, fallbacks):
    """Append per-τ rows to walk_forward_timesteps.csv (CSV-2 schema)."""
    out = []
    for r in rows:
        tau = r["Tau"]
        out.append({
            "Sweep":             sweep,
            "Seed":              SEED,
            "Tau":               tau,
            "N_labeled":         r["N_labeled"],
            "N_illicit":         r["N_illicit"],
            "N_licit":           r["N_licit"],
            "Low_Confidence":    r["Low_Confidence"],
            "Regime":            r["Regime"],
            "Train_Window_Size": train_window_sizes.get(tau, "N/A"),
            "Calib_Threshold":   round(float(thresholds.get(tau, "N/A")), 4)
                                  if tau in thresholds else "N/A",
            "Calib_Fallback":    fallbacks.get(tau, "N/A"),
            "F1":                r["F1"],
            "PRAUC":             r["PRAUC"],
            "Precision":         r["Precision"],
            "Recall":            r["Recall"],
            "Selfcond_Bug":      "fixed",
        })
    df_new = pd.DataFrame(out, columns=CSV2_COLS)
    if os.path.exists(TS_CSV):
        df_new.to_csv(TS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(TS_CSV, index=False)


def _write_csv1(sweep, agg, wf_time, feature_set, variation="Base"):
    """Append one aggregate row to sweep_results.csv (CSV-1 schema)."""
    df_new = pd.DataFrame([_make_result(
        seed=SEED, variation=variation, sweep=sweep,
        static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
        wf_time=round(wf_time, 2), wf_mem="N/A",
        wf_f1=agg["WF_Macro_F1"],          wf_prauc=agg["WF_Macro_PRAUC"],
        wf_pooled_f1=agg["WF_Pooled_F1"],  wf_pooled_prauc=agg["WF_Pooled_PRAUC"],
        wf_pre43_pooled_f1=agg["WF_Pre43_Pooled_F1"], wf_pre43_prauc=agg["WF_Pre43_PRAUC"],
        wf_shock_f1=agg["WF_Shock_F1"],    wf_shock_prauc=agg["WF_Shock_PRAUC"],
        wf_recovery_pooled_f1=agg["WF_Recovery_Pooled_F1"],
        wf_recovery_prauc=agg["WF_Recovery_PRAUC"],
        feature_set=feature_set,
        threshold_method="epsilon-fallback",
        selfcond_bug="fixed",
        notes="F4 exponential decay ablation, ε-fallback calib, one-step-ahead",
    )], columns=list(_RESULT_KEYS))
    if os.path.exists(SWEEP_CSV):
        df_new.to_csv(SWEEP_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(SWEEP_CSV, index=False)


# ─── XGBoost Logic ────────────────────────────────────────────────────────────

def compute_unified_xgb_weights(snapshots, labels, lambda_decay=0.25):
    """Exponential temporal decay × class-balance weights, mean-normalized."""
    tau_max     = np.max(snapshots)
    delta_t     = tau_max - snapshots
    temp_w      = np.exp(-lambda_decay * delta_t)

    neg_count   = np.sum(labels == 0)
    pos_count   = np.sum(labels == 1)
    class_mult  = neg_count / max(pos_count, 1)
    class_w     = np.where(labels == 1, class_mult, 1.0)

    combined    = temp_w * class_w
    return combined / np.mean(combined)   # normalize → effective N unchanged


def train_xgb_decay(dm, cfg, lambda_decay=0.25):
    """Walk-forward XGBoost with exponential decay sample weights.

    XGB hyper-parameters deliberately match F1 Base XGBoost (n_estimators=200,
    max_depth=6, lr=0.1, eval_metric='logloss').  Class balance is embedded in
    ``sample_weight`` so scale_pos_weight is intentionally omitted to avoid
    double-counting.
    """
    recs              = []
    train_window_sz   = {}
    thresholds        = {}
    fallbacks         = {}

    for tau in cfg.test_steps:
        tb          = list(range(1, tau))        # expanding train window [1 … τ-1]
        Xtr, ytr    = _tab_block(dm, tb, use_temporal=False)

        # Reconstruct per-row snapshot index (mirrors _tab_block)
        snaps = []
        for t in tb:
            g = dm.graphs[t]
            m = g["labeled_mask"].numpy()
            if m.sum() > 0:
                snaps.extend([t] * int(m.sum()))
        snaps = np.array(snaps)

        sample_weights = compute_unified_xgb_weights(snaps, ytr, lambda_decay)

        # ── SAME hyper-params as F1 Base XGBoost ──────────────────────────────
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss",
            random_state=cfg.seed, n_jobs=-1,
        )
        clf.fit(Xtr, ytr, sample_weight=sample_weights)

        Xte, yte = _tab_step(dm, tau, use_temporal=False)
        preds_proba = clf.predict_proba(Xte)[:, 1]

        # ε-fallback calibration on τ-1
        tau_cal     = tau - 1
        Xc, yc      = _tab_step(dm, tau_cal, use_temporal=False)
        cal_preds   = clf.predict_proba(Xc)[:, 1]
        illicit_rate = (yc == 1).mean()
        target_rate  = max(illicit_rate, 0.005)
        thresh       = np.percentile(cal_preds, 100 * (1 - target_rate))
        fallback     = illicit_rate < 0.005

        preds_bin = (preds_proba >= thresh).astype(int)

        recs.append({
            "tau":     tau,
            "y_true":  yte,
            "scores":  preds_proba,
            "y_pred":  preds_bin,
        })
        train_window_sz[tau] = len(tb)
        thresholds[tau]      = thresh
        fallbacks[tau]       = fallback

        print(f"    [xgb_decay] τ={tau} done", flush=True)

    return recs, train_window_sz, thresholds, fallbacks


# ─── PyTorch / SGC+MLP Logic ─────────────────────────────────────────────────

def get_batch_weights(batch_snapshots, batch_labels, lambda_decay, tau_max, class_mult):
    """Per-sample weights: temporal decay × class balance, batch-normalized."""
    delta_t  = tau_max - batch_snapshots
    temp_w   = torch.exp(-lambda_decay * delta_t.float())
    class_w  = torch.where(batch_labels == 1, float(class_mult), 1.0)
    combined = temp_w * class_w
    return combined / torch.mean(combined)


def fit_head_decay(Xtr, ytr, snapshots, in_dim, cfg, tau_max, lambda_decay=0.25):
    """Train one SGCHead with exponential temporal-decay weighted loss."""
    set_global_seeds(cfg.seed)
    Xtr       = Xtr.to(DEVICE)
    ytr       = ytr.to(DEVICE)
    snapshots = snapshots.to(DEVICE)

    model   = SGCHead(in_dim, cfg).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")   # MUST be none
    opt     = torch.optim.AdamW(
        model.parameters(), lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay,
    )

    mask = (ytr != -1)
    if mask.sum() == 0:
        return model

    Xtr_m       = Xtr[mask]
    ytr_m       = ytr[mask]
    snaps_m     = snapshots[mask]

    neg_count   = (ytr_m == 0).sum().item()
    pos_count   = (ytr_m == 1).sum().item()
    class_mult  = neg_count / max(pos_count, 1)
    
    # PRECOMPUTE: Compute static weights once before the training loop
    batch_w = get_batch_weights(snaps_m, ytr_m, lambda_decay, tau_max, class_mult)

    model.train()
    for _ in range(cfg.sgc_epochs):
        opt.zero_grad()
        logits      = model(Xtr_m)
        raw_loss    = loss_fn(logits, ytr_m)
        final_loss  = torch.mean(raw_loss * batch_w)

        if getattr(cfg, "sgc_l1_lambda", 0.0) > 0.0:
            l1_penalty = model.net[0].weight.abs().sum()
            final_loss = final_loss + cfg.sgc_l1_lambda * l1_penalty

        final_loss.backward()
        opt.step()

    return model


def train_sgc_decay(dm, cfg, lambda_decay=0.25):
    """Walk-forward SGC+MLP with exponential decay weighted loss.

    Uses K=2 clean (no PCA, no topo) as the anchor, matching the
    baseline used in F2 and F3 for mathematical rigor.
    """
    recs            = []
    train_window_sz = {}
    thresholds      = {}
    fallbacks       = {}

    for tau in cfg.test_steps:
        tb          = list(range(1, tau))
        Xs, ys, sn  = [], [], []
        for t in tb:
            g = dm.graphs[t]
            Xs.append(g["x"])
            ys.append(g["y"])
            sn.append(torch.full_like(g["y"], t))

        Xtr       = torch.cat(Xs)
        ytr       = torch.cat(ys)
        snapshots = torch.cat(sn)

        model = fit_head_decay(
            Xtr, ytr, snapshots,
            in_dim=Xtr.shape[1], cfg=cfg,
            tau_max=tau - 1,           # newest training step is τ-1 → weight=exp(0)=1
            lambda_decay=lambda_decay,
        )
        model.eval()

        Xte      = dm.graphs[tau]["x"].to(DEVICE)
        yte_full = dm.graphs[tau]["y"]
        mask_te  = dm.graphs[tau]["labeled_mask"]

        with torch.no_grad():
            preds_proba = torch.softmax(model(Xte[mask_te]), dim=-1)[:, 1].cpu().numpy()

        yte = yte_full[mask_te].numpy()

        # ε-fallback calibration on τ-1
        tau_cal  = tau - 1
        Xc       = dm.graphs[tau_cal]["x"].to(DEVICE)
        yc_full  = dm.graphs[tau_cal]["y"]
        mask_cal = dm.graphs[tau_cal]["labeled_mask"]

        with torch.no_grad():
            cal_preds = torch.softmax(model(Xc[mask_cal]), dim=-1)[:, 1].cpu().numpy()

        yc           = yc_full[mask_cal].numpy()
        illicit_rate = (yc == 1).mean()
        target_rate  = max(illicit_rate, 0.005)
        thresh       = np.percentile(cal_preds, 100 * (1 - target_rate))
        fallback     = illicit_rate < 0.005

        preds_bin = (preds_proba >= thresh).astype(int)

        recs.append({
            "tau":    tau,
            "y_true": yte,
            "scores": preds_proba,
            "y_pred": preds_bin,
        })
        train_window_sz[tau] = len(tb)
        thresholds[tau]      = thresh
        fallbacks[tau]       = fallback

        print(f"    [sgc_decay] τ={tau} done", flush=True)

    return recs, train_window_sz, thresholds, fallbacks


# ─── Run ──────────────────────────────────────────────────────────────────────

from data.load_dataset import download_and_load_data


def _run_and_write(sweep_name, train_fn, feature_set, variation="Base"):
    """Run one decay sweep, print summary, and persist results to both CSVs."""
    t0 = time.time()
    recs, win_sz, thresholds, fallbacks = train_fn()
    elapsed = time.time() - t0

    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(sweep_name, rows, win_sz, thresholds, fallbacks)
    _write_csv1(sweep_name, agg, elapsed, feature_set, variation=variation)

    print(
        f"  {sweep_name}: "
        f"Pooled_F1={agg['WF_Pooled_F1']}  "
        f"Recovery_PRAUC={agg['WF_Recovery_PRAUC']}  "
        f"Pooled_PRAUC={agg['WF_Pooled_PRAUC']}",
        flush=True,
    )
    return agg


def run():
    print("Loading raw dataset for F4...", flush=True)
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg = Config(
        train_steps=range(1, 27), val_steps=range(27, 35), test_steps=range(35, 50),
        sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
        use_graph_structural=False, use_directional_prop=False, seed=SEED,
    )

    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()

    print(f"[DataModule] feature_dim=165 | sgc_input_dim={dm.sgc_input_dim}", flush=True)

    lambdas = [0.05, 0.25, 0.50]

    # ── XGBoost decay grid ────────────────────────────────────────────────────
    print("\n=== F4: XGBoost Exponential Decay Grid ===", flush=True)
    xgb_results = {}
    for lam in lambdas:
        name = f"F4: XGBoost decay λ={lam}"
        # TEMPORARILY SKIPPING XGBOOST TO AVOID DUPLICATES IN CSV
        # print(f"\n--- {name} ---", flush=True)
        # agg = _run_and_write(
        #     sweep_name=name,
        #     train_fn=lambda lam=lam: train_xgb_decay(dm, cfg, lambda_decay=lam),
        #     feature_set="Raw-165 (no ts), exp-decay weights",
        #     variation="Base",
        # )
        # xgb_results[lam] = agg
        xgb_results[lam] = {"WF_Pooled_F1": "N/A", "WF_Recovery_PRAUC": "N/A"}

    # ── SGC+MLP decay grid ────────────────────────────────────────────────────
    print("\n=== F4: SGC+MLP Exponential Decay Grid ===", flush=True)
    sgc_results = {}
    for lam in lambdas:
        name = f"F4: SGC+MLP decay λ={lam}"
        print(f"\\n--- {name} ---", flush=True)
        agg = _run_and_write(
            sweep_name=name,
            train_fn=lambda lam=lam: train_sgc_decay(dm, cfg, lambda_decay=lam),
            feature_set=f"SGC K=2 multiscale + MLP ({dm.sgc_input_dim}-dim), exp-decay weights",
            variation="Base",
        )
        sgc_results[lam] = agg

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n=== FINAL ABLATION GRID RESULTS ===")
    print(f"{'Lambda':>6} | {'XGB Pooled F1':>14} | {'XGB Recov PRAUC':>15} "
          f"| {'SGC Pooled F1':>13} | {'SGC Recov PRAUC':>15}")
    print("-" * 72)
    for lam in lambdas:
        xr = xgb_results[lam]
        sr = sgc_results[lam]
        print(
            f" {lam:4.2f}  | {str(xr['WF_Pooled_F1']):>14} | {str(xr['WF_Recovery_PRAUC']):>15} "
            f"| {sr['WF_Pooled_F1']:>13.4f} | {sr['WF_Recovery_PRAUC']:>15.4f}"
        )


if __name__ == "__main__":
    run()
