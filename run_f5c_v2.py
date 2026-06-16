"""F5c-v2: re-baseline SGC+MLP K=2 with the timestep leak removed (feature_cols
now excludes 'ts'). If F1 drops >0.05 vs F5c (0.679, produced WITH ts), the
timestep was carrying load. Static OOT, train 1-34 / test 35-49, pooled.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import torch
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, _compute_class_weights
from evaluation.falsification_log import log_verdict
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")


def _append(result):
    df_new = pd.DataFrame([result], columns=list(_RESULT_KEYS))
    df = pd.concat([pd.read_csv(SWEEP_CSV, keep_default_na=False), df_new], ignore_index=True)
    df.to_csv(SWEEP_CSV, index=False)


def main():
    set_global_seeds(42)
    df, df_edge, _, feature_cols = download_and_load_data()
    assert "ts" not in feature_cols, "timestep leak not removed"
    print(f"feature_cols = {len(feature_cols)} (ts excluded)")

    cfg = Config(train_steps=range(1, 35), val_steps=range(35, 35), test_steps=range(35, 50),
                 sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
                 use_graph_structural=False, use_directional_prop=False, seed=42)
    t0 = time.time()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    Xtr, ytr = stack_prop(dm, list(cfg.train_steps))
    Xte, yte = stack_prop(dm, list(cfg.test_steps))
    cls_w = _compute_class_weights(ytr[ytr != -1], DEVICE)
    model = fit_head(Xtr, ytr, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    model.eval()
    with torch.no_grad():
        m = yte != -1
        s = torch.softmax(model(Xte[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    y = yte[m].numpy()
    f1 = round(float(f1_score(y, (s >= 0.5).astype(int), pos_label=1, zero_division=0)), 4)
    prauc = round(float(average_precision_score(y, s)), 4)

    _append(_make_result(seed=42, variation="Base", sweep="F5c-v2: SGC+MLP K=2 (no ts)",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=f1, static_prauc=prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=f"SGC K=2 multiscale + MLP, ts excluded ({dm.sgc_input_dim}-dim)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="F5c-v2: timestep leak removed; clean static baseline"))

    f5c_old = 0.6788
    delta = f1 - f5c_old
    carried = abs(delta) > 0.05
    print(f"  F5c    (with ts): F1=0.6788")
    print(f"  F5c-v2 (no ts)  : F1={f1:.4f}  PRAUC={prauc:.4f}  | delta={delta:+.4f}")
    print(f"  timestep carried load (|delta|>0.05): {carried}")

    log_verdict("F5c", "Corrected instrument: SGC+MLP K=2 (NOTE: F5c produced WITH ts leak)",
                World_Eliminated="broken-instrument (nonlinear)", Readout_Metric="Static_OOT_F1",
                Decision_Rule="SGC+MLP F1>=0.50 AND +0.15 over D1; superseded by F5c-v2 (ts excluded)",
                Observed_Value=f5c_old, Verdict="PASS",
                Sweep_Refs="F5c: SGC+MLP K=2 [corrected]",
                Notes="Produced WITH timestep in features (leak). See F5c-v2 for clean baseline.")
    log_verdict("F5c-v2", "Clean instrument: SGC+MLP K=2, timestep excluded",
                World_Eliminated="broken-instrument (nonlinear)", Readout_Metric="Static_OOT_F1",
                Decision_Rule="F1>=0.50 (clean baseline, no ts leak)",
                Observed_Value=f1, Verdict="PASS" if f1 >= 0.50 else "FAIL",
                Sweep_Refs="F5c-v2: SGC+MLP K=2 (no ts)",
                Notes=f"PRAUC={prauc}; delta vs F5c(with ts)={delta:+.4f}; ts carried load={carried}")


if __name__ == "__main__":
    main()
