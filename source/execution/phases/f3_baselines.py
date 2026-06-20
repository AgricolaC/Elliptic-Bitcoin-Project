"""Phase 1 — F3 instrument check (Clean static baseline).

Validates the measurement apparatus against published Elliptic baselines and tests 
the non-linear graph architecture (SGC K=2 + MLP) on the standard temporal split 
(train τ=1..34, test τ=35..49) without any data leakage (timestep 'ts' excluded).
It also runs a PyG GCN reference model to confirm graph structural signal.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(os.path.dirname(HERE))
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import _compute_class_weights, fit_head, stack_prop
from evaluation.falsification_log import log_verdict
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")
TRAIN = range(1, 35)
TEST = range(35, 50)

def _pooled_static(scores, y_true):
    f1 = float(f1_score(y_true, (scores >= 0.5).astype(int), pos_label=1, zero_division=0))
    prauc = float(average_precision_score(y_true, scores))
    return round(f1, 4), round(prauc, 4)

def _append(result: dict):
    df_new = pd.DataFrame([result], columns=list(_RESULT_KEYS))
    if os.path.exists(SWEEP_CSV):
        df = pd.read_csv(SWEEP_CSV, keep_default_na=False)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(SWEEP_CSV, index=False)

class GCN2(nn.Module):
    def __init__(self, in_dim, hidden, n_classes=2):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, n_classes)

    def forward(self, x, edge_index):
        h = torch.relu(self.c1(x, edge_index))
        return self.c2(h, edge_index)

def run_gcn(dm, device, epochs=1000, hidden=100):
    set_global_seeds(42)
    in_dim = dm.graphs[min(dm.graphs)]["x"].shape[1]
    model = GCN2(in_dim, hidden).to(device)
    ytr = torch.cat([dm.graphs[t]["y"][dm.graphs[t]["labeled_mask"]] for t in TRAIN if t in dm.graphs])
    cls_w = torch.tensor([0.3, 0.7], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    edges = {t: to_undirected(dm.graphs[t]["edge_index"]).to(device) for t in dm.graphs}
    for _ in range(epochs):
        opt.zero_grad()
        total = 0.0
        for t in TRAIN:
            if t not in dm.graphs:
                continue
            g = dm.graphs[t]
            m = g["labeled_mask"]
            if m.sum() == 0:
                continue
            logits = model(g["x"].to(device), edges[t])
            total = total + loss_fn(logits[m], g["y"][m].to(device))
        total.backward()
        opt.step()

    model.eval()
    s_all, y_all = [], []
    with torch.no_grad():
        for t in TEST:
            if t not in dm.graphs:
                continue
            g = dm.graphs[t]
            m = g["labeled_mask"]
            if m.sum() == 0:
                continue
            logits = model(g["x"].to(device), edges[t])
            s_all.append(torch.softmax(logits[m], dim=1)[:, 1].cpu().numpy())
            y_all.append(g["y"][m].numpy())
    return _pooled_static(np.concatenate(s_all), np.concatenate(y_all))

def run():
    set_global_seeds(42)
    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()
    
    assert "ts" not in feature_cols, "timestep leak not removed"

    feat_cols = [c for c in df.columns if c not in ("txId", "ts", "label", "class")]
    tr = df[df.ts.between(1, 34)]
    te = df[df.ts.between(35, 49)]
    mtr, mte = tr.label != -1, te.label != -1
    Xtr_tab, ytr_tab = tr.loc[mtr, feat_cols].values, tr.loc[mtr, "label"].values
    Xte_tab, yte_tab = te.loc[mte, feat_cols].values, te.loc[mte, "label"].values
    n_neg, n_pos = int((ytr_tab == 0).sum()), int((ytr_tab == 1).sum())
    spw = n_neg / max(n_pos, 1)

    feat_desc = f"Raw-{len(feat_cols)} (no ts)"

    # ── F3a: Base XGBoost ─────────────────────────────────────────────────────
    t0 = time.time()
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=spw, eval_metric="logloss",
                        n_jobs=-1, random_state=42)
    xgb.fit(Xtr_tab, ytr_tab)
    xgb_f1, xgb_prauc = _pooled_static(xgb.predict_proba(Xte_tab)[:, 1], yte_tab)
    _append(_make_result(seed=42, variation="Base", sweep="F3a: Base XGBoost (clean)",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=xgb_f1, static_prauc=xgb_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=feat_desc, threshold_method="fixed-0.5",
                         selfcond_bug="fixed", notes="F3 instrument check, static OOT pooled 35-49"))
    print(f"  [F3a] Base XGBoost   F1={xgb_f1:.4f}  PRAUC={xgb_prauc:.4f}")

    # ── F3b: Random Forest ────────────────────────────────────────────────────
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(Xtr_tab, ytr_tab)
    rf_f1, rf_prauc = _pooled_static(rf.predict_proba(Xte_tab)[:, 1], yte_tab)
    _append(_make_result(seed=42, variation="Base", sweep="F3b: Random Forest (clean)",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=rf_f1, static_prauc=rf_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=feat_desc, threshold_method="fixed-0.5",
                         selfcond_bug="fixed", notes="F3 instrument check, static OOT pooled 35-49"))
    print(f"  [F3b] Random Forest  F1={rf_f1:.4f}  PRAUC={rf_prauc:.4f}")

    # ── F3c: SGC K=2 multiscale + MLP (Nonlinear Graph) ───────────────────────
    cfg = Config(train_steps=TRAIN, val_steps=range(35, 35), test_steps=TEST,
                 sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
                 use_graph_structural=False, use_directional_prop=False, seed=42)
    t0 = time.time()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    Xtr_g, ytr_g = stack_prop(dm, list(TRAIN))
    Xte_g, yte_g = stack_prop(dm, list(TEST))
    cls_w = _compute_class_weights(ytr_g[ytr_g != -1], DEVICE)
    model = fit_head(Xtr_g, ytr_g, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    model.eval()
    with torch.no_grad():
        m = yte_g != -1
        s = torch.softmax(model(Xte_g[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    sgc_f1, sgc_prauc = _pooled_static(s, yte_g[m].numpy())
    
    _append(_make_result(seed=42, variation="Base", sweep="F3c: SGC+MLP K=2 (clean)",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=sgc_f1, static_prauc=sgc_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=f"SGC K=2 multiscale + MLP ({dm.sgc_input_dim}-dim, no ts)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="F3 instrument check (clean baseline), static OOT pooled 35-49"))
    print(f"  [F3c] SGC+MLP K=2    F1={sgc_f1:.4f}  PRAUC={sgc_prauc:.4f}")

    # ── F3e: sklearn Logistic Regression ──────────────────────────────────────
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(Xtr_tab, ytr_tab)
    lr_f1, lr_prauc = _pooled_static(lr.predict_proba(Xte_tab)[:, 1], yte_tab)
    _append(_make_result(seed=42, variation="Base", sweep="Diagnostic: sklearn LR",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=lr_f1, static_prauc=lr_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=feat_desc, threshold_method="fixed-0.5",
                         selfcond_bug="fixed", notes="Linear baseline ceiling"))
    print(f"  [F3e] sklearn LR     F1={lr_f1:.4f}  PRAUC={lr_prauc:.4f}")

    # ── F3d: GCN 2-layer reference ────────────────────────────────────────────
    t0 = time.time()
    gcn_device = torch.device("cpu")
    gcn_f1, gcn_prauc = run_gcn(dm, gcn_device)
    _append(_make_result(seed=42, variation="Base", sweep="F3d: GCN reference [2-layer]",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=gcn_f1, static_prauc=gcn_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set="2-layer GCNConv (PyG, undirected, hidden=100)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="F3/F5 GCN reference (Weber et al), static OOT pooled 35-49"))
    print(f"  [F3d] GCN 2-layer    F1={gcn_f1:.4f}  PRAUC={gcn_prauc:.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    pass_xgb = xgb_f1 >= 0.74
    pass_rf = rf_f1 >= 0.79
    pass_sgc = sgc_f1 >= 0.50
    pass_gcn = 0.50 <= gcn_f1 <= 0.75

    verdict = "PASS" if (pass_xgb and pass_rf and pass_sgc and pass_gcn) else "FAIL"
    notes = (f"XGB F1={xgb_f1:.4f}(>=0.74); RF F1={rf_f1:.4f}(>=0.79); "
             f"SGC+MLP F1={sgc_f1:.4f}(>=0.50); GCN F1={gcn_f1:.4f}([0.5,0.75]); "
             f"LR F1={lr_f1:.4f}")
    
    log_verdict("F3", "Instrument check (clean baselines & graph reference)",
                World_Eliminated="broken-instrument", Readout_Metric="Static_OOT_F1",
                Decision_Rule="XGB>=0.74 & RF>=0.79 & SGC>=0.50 & GCN in [0.5,0.75]",
                Observed_Value=round(sgc_f1, 4), Verdict=verdict,
                Sweep_Refs="F3a: XGB, F3b: RF, F3c: SGC+MLP, F3d: GCN, F3e: LR",
                Notes=notes)

    print("\n" + ("=" * 60))
    if verdict == "PASS":
        print("F3 PASS — clean instrument confirmed. Ready for walk-forward runs.")
    else:
        broken = [n for n, ok in [("XGBoost", pass_xgb), ("RF", pass_rf), 
                                  ("SGC+MLP", pass_sgc), ("GCN", pass_gcn)] if not ok]
        print(f"F3 FAIL — instrument suspect at: {', '.join(broken)}. STOPPING. {notes}")
    print("=" * 60)

if __name__ == "__main__":
    run()
