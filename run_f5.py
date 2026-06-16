"""Phase 1 — F5 instrument check (static OOT only, no walk-forward).

Validates the measurement apparatus against published Elliptic baselines before
any comparative reading. Three sub-runs on the standard temporal split
(train τ=1..34, test τ=35..49, pooled, labeled nodes only):

  a. Base XGBoost (165 raw numeric features)   PASS: Static_OOT_F1 >= 0.74
  b. Random Forest (165 raw numeric features)  PASS: Static_OOT_F1 >= 0.79
  c. SGC K=1 linear, no MLP, no topo           PASS: 0.55 <= F1 < RF result

Rows are appended to results/sweep_results.csv under the v2 schema with
Selfcond_Bug="fixed". The F5 verdict is written to results/falsification_log.csv.
Stops after F5 — does not run walk-forward.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, _compute_class_weights
from evaluation.falsification_log import log_verdict
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")

# Old -> new column rename for migrating any pre-fix rows.
_RENAME = {
    "Feature Set": "Feature_Set", "Threshold": "Threshold_Method",
    "Static Time (s)": "Static_Time_s", "Static Mem (MB)": "Static_Mem_MB",
    "Static Val F1": "Static_Val_F1", "Static Val PR-AUC": "Static_Val_PRAUC",
    "Static OOT F1": "Static_OOT_F1", "Static OOT PR-AUC": "Static_OOT_PRAUC",
    "WF Time (s)": "WF_Time_s", "WF Mem (MB)": "WF_Mem_MB",
    "Walk-Forward Mean F1": "WF_Macro_F1", "Walk-Forward Mean PR-AUC": "WF_Macro_PRAUC",
}


def _migrate_existing():
    """Migrate any pre-fix sweep_results.csv rows to the v2 schema in place."""
    if not os.path.exists(SWEEP_CSV):
        return
    df = pd.read_csv(SWEEP_CSV, keep_default_na=False)
    if "Selfcond_Bug" in df.columns:
        return  # already migrated
    df = df.rename(columns=_RENAME)
    for col in _RESULT_KEYS:
        if col not in df.columns:
            df[col] = "N/A"
    df["Selfcond_Bug"] = "present"
    df["Notes"] = "migrated from pre-fix schema"
    df = df[list(_RESULT_KEYS)]
    df.to_csv(SWEEP_CSV, index=False)
    print(f"Migrated {len(df)} pre-fix rows to v2 schema.")


def _append(result: dict):
    df_new = pd.DataFrame([result], columns=list(_RESULT_KEYS))
    if os.path.exists(SWEEP_CSV):
        df = pd.read_csv(SWEEP_CSV, keep_default_na=False)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(SWEEP_CSV, index=False)


def _pooled_static(scores, y_true):
    f1 = float(f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0))
    prauc = float(average_precision_score(y_true, scores))
    return round(f1, 4), round(prauc, 4)


def main():
    set_global_seeds(42)
    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    feat_cols = [c for c in df.columns if c not in ("txId", "ts", "label", "class")]
    print(f"Tabular feature count: {len(feat_cols)} (excludes txId/ts/label/class)")

    tr = df[df.ts.between(1, 34)]
    te = df[df.ts.between(35, 49)]
    mtr, mte = tr.label != -1, te.label != -1
    Xtr, ytr = tr.loc[mtr, feat_cols].values, tr.loc[mtr, "label"].values
    Xte, yte = te.loc[mte, feat_cols].values, te.loc[mte, "label"].values
    n_neg, n_pos = int((ytr == 0).sum()), int((ytr == 1).sum())
    spw = n_neg / max(n_pos, 1)

    _migrate_existing()
    feat_desc = f"Raw-{len(feat_cols)} (no ts)"

    # ── F5a: Base XGBoost ─────────────────────────────────────────────────────
    t0 = time.time()
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=spw, eval_metric="logloss",
                        n_jobs=-1, random_state=42)
    xgb.fit(Xtr, ytr)
    xgb_f1, xgb_prauc = _pooled_static(xgb.predict_proba(Xte)[:, 1], yte)
    _append(_make_result(seed=42, variation="Base", sweep="F5: Base XGBoost [v2-fixed]",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=xgb_f1, static_prauc=xgb_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=feat_desc, threshold_method="fixed-0.5",
                         selfcond_bug="fixed", notes="F5 instrument check, static OOT pooled 35-49"))
    print(f"  [F5a] Base XGBoost   F1={xgb_f1:.4f}  PRAUC={xgb_prauc:.4f}")

    # ── F5b: Random Forest ────────────────────────────────────────────────────
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    rf_f1, rf_prauc = _pooled_static(rf.predict_proba(Xte)[:, 1], yte)
    _append(_make_result(seed=42, variation="Base", sweep="F5: Random Forest [v2-fixed]",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=rf_f1, static_prauc=rf_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=feat_desc, threshold_method="fixed-0.5",
                         selfcond_bug="fixed", notes="F5 instrument check, static OOT pooled 35-49"))
    print(f"  [F5b] Random Forest  F1={rf_f1:.4f}  PRAUC={rf_prauc:.4f}")

    # ── F5c: SGC K=1 linear, no MLP, no topo ──────────────────────────────────
    t0 = time.time()
    cfg = Config(train_steps=range(1, 35), val_steps=range(35, 35), test_steps=range(35, 50),
                 sgc_k=1, use_multiscale_prop=False, use_mlp_head=False,
                 use_graph_structural=False, use_directional_prop=False)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
    Xte_g, yte_g = stack_prop(dm, list(cfg.test_steps))
    cls_w = _compute_class_weights(ytr_g[ytr_g != -1], DEVICE)
    model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    model.eval()
    with torch.no_grad():
        m = yte_g != -1
        s = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    sgc_f1, sgc_prauc = _pooled_static(s, yte_g[m].numpy())
    _append(_make_result(seed=42, variation="Base", sweep="F5: SGC K=1 linear [v2-fixed]",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=sgc_f1, static_prauc=sgc_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=f"SGC K=1 (S·X, {dm.sgc_input_dim}-dim incl ts)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="F5 instrument check, static OOT pooled 35-49"))
    print(f"  [F5c] SGC K=1 linear F1={sgc_f1:.4f}  PRAUC={sgc_prauc:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    pass_xgb = xgb_f1 >= 0.74
    pass_rf = rf_f1 >= 0.79
    pass_sgc = 0.55 <= sgc_f1 < rf_f1
    verdict = "PASS" if (pass_xgb and pass_rf and pass_sgc) else "FAIL"
    notes = (f"XGB F1={xgb_f1:.4f}(>=0.74:{pass_xgb}); RF F1={rf_f1:.4f}(>=0.79:{pass_rf}); "
             f"SGC F1={sgc_f1:.4f}(in[0.55,RF):{pass_sgc}). Tabular={len(feat_cols)} feats (no ts).")
    log_verdict("F5", "Instrument check vs published Elliptic baselines",
                World_Eliminated="broken-instrument", Readout_Metric="Static_OOT_F1",
                Decision_Rule="XGB>=0.74 AND RF>=0.79 AND 0.55<=SGC<RF",
                Observed_Value=round(xgb_f1, 4), Verdict=verdict,
                Sweep_Refs="F5: Base XGBoost [v2-fixed], F5: Random Forest [v2-fixed], F5: SGC K=1 linear [v2-fixed]",
                Notes=notes)

    print("\n" + ("=" * 60))
    if verdict == "PASS":
        print("F5 PASS — instrument confirmed. Ready for F1 walk-forward runs.")
    else:
        broken = [n for n, ok in [("XGBoost", pass_xgb), ("RF", pass_rf), ("SGC", pass_sgc)] if not ok]
        print(f"F5 FAIL — instrument suspect at: {', '.join(broken)}. STOPPING. {notes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
