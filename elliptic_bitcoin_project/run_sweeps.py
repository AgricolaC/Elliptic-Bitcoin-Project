"""
run_sweeps.py — Ablation sweep runner.

W4 FIX: Expanded ablation matrix that isolates each mechanism independently.
W5 FIX: All result dicts use the standardized key schema:
         {"Sweep", "Static OOT F1", "Static OOT PR-AUC",
          "Walk-Forward Mean F1", "Walk-Forward Mean PR-AUC"}
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, walk_forward_validation

warnings.filterwarnings("ignore", category=UserWarning)

# W5 FIX: canonical key set used everywhere
_RESULT_KEYS = (
    "Sweep",
    "Static OOT F1",
    "Static OOT PR-AUC",
    "Walk-Forward Mean F1",
    "Walk-Forward Mean PR-AUC",
)


def _make_result(
    sweep: str,
    static_f1: float | str,
    static_prauc: float | str,
    wf_f1: float | str,
    wf_prauc: float | str,
) -> dict:
    """
    Construct a result dict with the standardized key schema (W5 fix).
    Raises AssertionError if any key would be missing.
    """
    result = {
        "Sweep":                    sweep,
        "Static OOT F1":            static_f1,
        "Static OOT PR-AUC":        static_prauc,
        "Walk-Forward Mean F1":     wf_f1,
        "Walk-Forward Mean PR-AUC": wf_prauc,
    }
    # SHAPE GUARD: verify key completeness on every call
    assert set(result.keys()) == set(_RESULT_KEYS), \
        f"Result key schema violation: {set(result.keys())} != {set(_RESULT_KEYS)}"
    return result


def run_single_sweep(
    name: str,
    cfg: Config,
    df: pd.DataFrame,
    df_edge: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Run one full sweep (static OOT + walk-forward) and return a standardized result dict.

    W5 FIX: Returns _make_result(...) which enforces the canonical key schema.
    W6 FIX: dm.setup() now runs propagation internally; no external assignment needed.
    W7 FIX: Passes sweep name to walk_forward_validation for unique plot filenames.
    W8 FIX: Class weights are computed inside walk_forward_validation per tau.
    """
    set_global_seeds(cfg.seed)

    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    # W6: sgc_input_dim and 'prop' keys are now set inside setup() — no manual step.

    Xtr_g, ytr_g = stack_prop(dm, list(cfg.train_steps))
    Xte_g, yte_g = stack_prop(dm, list(cfg.test_steps))

    # Static OOT class weights (used only for the static evaluation head)
    valid_ytr = ytr_g[ytr_g != -1]
    counts    = torch.bincount(valid_ytr, minlength=2).float().clamp(min=1.0)
    cls_w     = (counts.sum() / (2.0 * counts)).to(DEVICE)

    model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    model.eval()
    with torch.no_grad():
        m      = (yte_g != -1)
        scores = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()

    y_true       = yte_g[m].numpy()
    static_f1    = f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)
    static_prauc = average_precision_score(y_true, scores)

    # W7: sweep_name embedded in filename; W8: cls_w computed inside per tau
    wf_f1, wf_prauc = walk_forward_validation(dm, cfg, DEVICE, sweep_name=name)

    return _make_result(
        sweep=name,
        static_f1=round(static_f1, 3),
        static_prauc=round(static_prauc, 3),
        wf_f1=round(wf_f1, 3),
        wf_prauc=round(wf_prauc, 3),
    )


def main():
    from data.load_dataset import download_and_load_data
    from analysis.eda import plot_temporal_distribution

    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    print("\n--- Phase 1: EDA ---")
    cfg_default = Config()
    plot_temporal_distribution(df, cfg_default)

    results = []

    # ── Baselines (tabular, no GNN) ────────────────────────────────────────────
    print("\n--- Baseline Tabular Models ---")
    try:
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        dm_base = EllipticDataModule(df, df_edge, feature_cols, cfg_default)
        dm_base.setup()

        Xs_tr, ys_tr = [], []
        for t in cfg_default.train_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_tr.append(g["x"].numpy()[:, :166][m])
            ys_tr.append(g["y"].numpy()[m])
        Xtr_b, ytr_b = np.concatenate(Xs_tr), np.concatenate(ys_tr)

        Xs_te, ys_te = [], []
        for t in cfg_default.test_steps:
            g = dm_base.graphs[t]; m = g["labeled_mask"].numpy()
            Xs_te.append(g["x"].numpy()[:, :166][m])
            ys_te.append(g["y"].numpy()[m])
        Xte_b, yte_b = np.concatenate(Xs_te), np.concatenate(ys_te)

        spw = (ytr_b == 0).sum() / max((ytr_b == 1).sum(), 1)
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="aucpr",
                             random_state=cfg_default.seed, n_jobs=1).fit(Xtr_b, ytr_b)
        s_xgb = xgb.predict_proba(Xte_b)[:, 1]
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    n_jobs=1, random_state=cfg_default.seed).fit(Xtr_b, ytr_b)
        s_rf = rf.predict_proba(Xte_b)[:, 1]

        results.append(_make_result(
            "Baseline: XGBoost (166)",
            round(f1_score(yte_b, (s_xgb >= 0.5).astype(int), pos_label=1), 3),
            round(average_precision_score(yte_b, s_xgb), 3),
            "N/A", "N/A",
        ))
        results.append(_make_result(
            "Baseline: RandomForest (166)",
            round(f1_score(yte_b, (s_rf >= 0.5).astype(int), pos_label=1), 3),
            round(average_precision_score(yte_b, s_rf), 3),
            "N/A", "N/A",
        ))
    except Exception as e:
        print(f"  Baselines skipped: {e}")

    # ── W4 FIX: Expanded ablation matrix ──────────────────────────────────────
    # Each sweep toggles exactly ONE mechanism relative to the previous row,
    # enabling unambiguous attribution of gain.
    sweeps = [
        # name                          use_mlp  use_ms   use_topo  use_recon  use_focal
        ("Sweep 1: SGC (baseline)",
         Config(use_mlp_head=False, use_multiscale_prop=False,
                use_topology=False, use_recon_error=False, use_focal_loss=False)),

        ("Sweep 2: + MLP Head",
         Config(use_mlp_head=True,  use_multiscale_prop=False,
                use_topology=False, use_recon_error=False, use_focal_loss=False)),

        ("Sweep 3: + Multiscale Prop",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=False, use_recon_error=False, use_focal_loss=False)),

        ("Sweep 4a: + Topology only",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=True,  use_recon_error=False, use_focal_loss=False)),

        ("Sweep 4b: + Recon Error only",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=False, use_recon_error=True,  use_focal_loss=False)),

        ("Sweep 5: + Full Self-Supervision",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=True,  use_recon_error=True,  use_focal_loss=False)),

        ("Sweep 6: + Focal Loss (vs Weighted CE)",
         Config(use_mlp_head=True,  use_multiscale_prop=True,
                use_topology=True,  use_recon_error=True,  use_focal_loss=True)),
    ]

    for name, cfg in sweeps:
        print(f"\n{'='*55}\nRunning: {name}\n{'='*55}")
        res = run_single_sweep(name, cfg, df, df_edge, feature_cols)
        results.append(res)
        
        # Incremental save
        pd.DataFrame(results, columns=list(_RESULT_KEYS)).to_csv(os.path.join(OUTPUT_DIR, "sweep_results.csv"), index=False)
        print(f"--> {res}\n")

    # ── Advanced modules ───────────────────────────────────────────────────────
    cfg_full = sweeps[-1][1]
    dm_adv   = EllipticDataModule(df, df_edge, feature_cols, cfg_full)
    dm_adv.setup()

    try:
        from models.pu_learning import pu_learning_adjust
        res_pu = pu_learning_adjust(dm_adv, cfg_full)
        # Normalize keys (advanced modules use Static OOT F1 already)
        results.append(_make_result(
            res_pu["Sweep"],
            res_pu.get("Static OOT F1", "N/A"),
            res_pu.get("Static OOT PR-AUC", "N/A"),
            res_pu.get("Walk-Forward Mean F1", "N/A"),
            res_pu.get("Walk-Forward Mean PR-AUC", "N/A"),
        ))
    except Exception as e:
        print(f"  PU Learning skipped: {e}")

    try:
        from models.drift_adaptation import explicit_drift_adaptation
        res_drift = explicit_drift_adaptation(dm_adv, cfg_full)
        results.append(_make_result(
            res_drift["Sweep"],
            res_drift.get("Static OOT F1", "N/A"),
            res_drift.get("Static OOT PR-AUC", "N/A"),
            res_drift.get("Walk-Forward Mean F1", "N/A"),
            res_drift.get("Walk-Forward Mean PR-AUC", "N/A"),
        ))
    except Exception as e:
        print(f"  Drift adaptation skipped: {e}")

    try:
        from models.stacking import stacking_meta_classifier
        res_stack = stacking_meta_classifier(dm_adv, cfg_full)
        results.append(_make_result(
            res_stack["Sweep"],
            res_stack.get("Static OOT F1", "N/A"),
            res_stack.get("Static OOT PR-AUC", "N/A"),
            res_stack.get("Walk-Forward Mean F1", "N/A"),
            res_stack.get("Walk-Forward Mean PR-AUC", "N/A"),
        ))
    except Exception as e:
        print(f"  Stacking skipped: {e}")

    # ── Persist results ────────────────────────────────────────────────────────
    df_res   = pd.DataFrame(results, columns=list(_RESULT_KEYS))
    out_file = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    df_res.to_csv(out_file, index=False)
    print(f"\nResults saved to {out_file}")

    # Verify no NaN from key mismatch (W5 guard)
    nan_cols = df_res.columns[df_res.isna().any()].tolist()
    assert not nan_cols, (
        f"W5 GUARD: NaN columns found in sweep_results.csv: {nan_cols}. "
        f"This indicates a key schema mismatch — check _make_result usage."
    )

    print("\n--- FINAL ABLATION RESULTS ---")
    for r in results:
        print(
            f"{r['Sweep']:40s} | "
            f"Static F1={r['Static OOT F1']} | "
            f"PR-AUC={r['Static OOT PR-AUC']} | "
            f"WF F1={r['Walk-Forward Mean F1']} | "
            f"WF PR-AUC={r['Walk-Forward Mean PR-AUC']}"
        )


if __name__ == "__main__":
    main()
