"""SGC instrument root-cause diagnostic (no fixes, no battery verdict).

D1 — K=0 control: raw X through the SAME scaler_prop + fit_head linear path as
     F5c, only without propagation. Isolates propagation vs head/training.
D2 — sklearn LogisticRegression on standardized raw 165 features (external
     reference, Weber LR ≈ 0.45). Isolates fit_head vs a plain linear model.

Split standardized to F5's train τ=1..34 / test τ=35..49 (NOT the 1..26 written
in the D2 spec) so D1, D2, F5a/b/c are all directly comparable — same train data,
only the model/representation differs.

Reports four numbers and appends two rows to sweep_results.csv. Writes NO
falsification_log verdict (diagnostic, not battery). Applies no fix.
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, _compute_class_weights
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")


def _pooled(scores, y):
    return (round(float(f1_score(y, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)), 4),
            round(float(average_precision_score(y, scores)), 4))


def _append(result):
    df_new = pd.DataFrame([result], columns=list(_RESULT_KEYS))
    df = pd.concat([pd.read_csv(SWEEP_CSV, keep_default_na=False), df_new], ignore_index=True) \
        if os.path.exists(SWEEP_CSV) else df_new
    df.to_csv(SWEEP_CSV, index=False)


def main():
    set_global_seeds(42)
    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    # ── D1: K=0 (raw X) through scaler_prop + fit_head linear ─────────────────
    t0 = time.time()
    cfg = Config(train_steps=range(1, 35), val_steps=range(35, 35), test_steps=range(35, 50),
                 sgc_k=0, use_multiscale_prop=False, use_mlp_head=False,
                 use_graph_structural=False, use_directional_prop=False)
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
    d1_f1, d1_prauc = _pooled(s, yte[m].numpy())
    _append(_make_result(seed=42, variation="Base", sweep="Diagnostic: K=0 linear",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=d1_f1, static_prauc=d1_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=f"raw X via scaler_prop+fit_head ({dm.sgc_input_dim}-dim)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="Diagnostic D1: K=0 control vs F5c K=1"))
    print(f"  [D1] K=0 linear (fit_head)  F1={d1_f1:.4f}  PRAUC={d1_prauc:.4f}")

    # ── D2: sklearn LogisticRegression on standardized raw 165 ────────────────
    t0 = time.time()
    feat = [c for c in df.columns if c not in ("txId", "ts", "label", "class")]
    tr = df[df.ts.between(1, 34)]; te = df[df.ts.between(35, 49)]
    mtr, mte = tr.label != -1, te.label != -1
    Xtr2, ytr2 = tr.loc[mtr, feat].values, tr.loc[mtr, "label"].values
    Xte2, yte2 = te.loc[mte, feat].values, te.loc[mte, "label"].values
    sc = StandardScaler().fit(Xtr2)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    lr.fit(sc.transform(Xtr2), ytr2)
    s2 = lr.predict_proba(sc.transform(Xte2))[:, 1]
    d2_f1, d2_prauc = _pooled(s2, yte2)
    _append(_make_result(seed=42, variation="Base", sweep="Diagnostic: sklearn LR",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=d2_f1, static_prauc=d2_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=f"Raw-{len(feat)} standardized (no ts)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="Diagnostic D2: sklearn LR external reference"))
    print(f"  [D2] sklearn LR             F1={d2_f1:.4f}  PRAUC={d2_prauc:.4f}")

    print("\n--- Diagnostic summary (vs F5c SGC K=1 = 0.247/0.213) ---")
    print(f"  D1 K=0 linear (fit_head): F1={d1_f1:.4f}  PRAUC={d1_prauc:.4f}")
    print(f"  D2 sklearn LR           : F1={d2_f1:.4f}  PRAUC={d2_prauc:.4f}")


if __name__ == "__main__":
    main()
