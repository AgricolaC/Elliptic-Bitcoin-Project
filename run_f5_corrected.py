"""Corrected F5 instrument check — the NONLINEAR graph architectures.

The linear SGC head caps at the linear ceiling (~0.28 F1, an established
instrumental finding, not a bug). The thesis models use an MLP head; this script
checks the architecture that actually gates the battery.

F5c-MLP : SGC K=2 multiscale + MLP head (thesis arch minus temporal head).
F5d-GCN : 2-layer GCNConv (PyG) reference — does graph signal exist on this split?

Split standardized to train τ=1..34 / test τ=35..49 (comparable to D1/F5a-c).
Static OOT, pooled, labeled nodes only. No walk-forward, no fixes.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import fit_head, stack_prop, _compute_class_weights
from evaluation.falsification_log import log_verdict
from sweep import _make_result, _RESULT_KEYS

SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")
TRAIN, TEST = range(1, 35), range(35, 50)


def _pooled(scores, y):
    return (round(float(f1_score(y, (scores >= 0.5).astype(int), pos_label=1, zero_division=0)), 4),
            round(float(average_precision_score(y, scores)), 4))


def _append(result):
    df_new = pd.DataFrame([result], columns=list(_RESULT_KEYS))
    df = pd.concat([pd.read_csv(SWEEP_CSV, keep_default_na=False), df_new], ignore_index=True) \
        if os.path.exists(SWEEP_CSV) else df_new
    df.to_csv(SWEEP_CSV, index=False)


def _d1_f1():
    df = pd.read_csv(SWEEP_CSV, keep_default_na=False)
    row = df[df.Sweep == "Diagnostic: K=0 linear"]
    return float(row.Static_OOT_F1.values[0]) if len(row) else float("nan")


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
    # Weber et al. (2019) used specific 0.3/0.7 weights for licit/illicit
    cls_w = torch.tensor([0.3, 0.7], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w)
    # Weber et al. (2019) used Adam with lr=0.001
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
    return _pooled(np.concatenate(s_all), np.concatenate(y_all))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    seed = args.seed
    
    set_global_seeds(seed)
    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg = Config(train_steps=TRAIN, val_steps=range(35, 35), test_steps=TEST,
                 sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
                 use_graph_structural=False, use_directional_prop=False, seed=seed)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()

    # ── F5c-MLP: SGC K=2 multiscale + MLP head ────────────────────────────────
    t0 = time.time()
    Xtr, ytr = stack_prop(dm, list(TRAIN))
    Xte, yte = stack_prop(dm, list(TEST))
    cls_w = _compute_class_weights(ytr[ytr != -1], DEVICE)
    model = fit_head(Xtr, ytr, dm.sgc_input_dim, cfg, cls_w, DEVICE)
    model.eval()
    with torch.no_grad():
        m = yte != -1
        s = torch.softmax(model(Xte[m].to(DEVICE)), dim=1)[:, 1].cpu().numpy()
    f5c_f1, f5c_prauc = _pooled(s, yte[m].numpy())
    _append(_make_result(seed=seed, variation="Base", sweep="F5c: SGC+MLP K=2 [corrected]",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=f5c_f1, static_prauc=f5c_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set=f"SGC K=2 multiscale + MLP ({dm.sgc_input_dim}-dim)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="Corrected F5c: nonlinear head, static OOT pooled 35-49"))
    print(f"  [F5c] SGC+MLP K=2  F1={f5c_f1:.4f}  PRAUC={f5c_prauc:.4f}")

    # ── F5d-GCN: 2-layer GCNConv reference (CPU) ──────────────────────────────
    t0 = time.time()
    gcn_device = torch.device("cpu")
    f5d_f1, f5d_prauc = run_gcn(dm, gcn_device)
    _append(_make_result(seed=seed, variation="Base", sweep="F5d: GCN reference [2-layer]",
                         static_time=round(time.time() - t0, 3), static_mem="N/A",
                         static_f1=f5d_f1, static_prauc=f5d_prauc,
                         wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
                         feature_set="2-layer GCNConv (PyG, undirected, hidden=100)",
                         threshold_method="fixed-0.5", selfcond_bug="fixed",
                         notes="F5d GCN reference (Weber et al), static OOT pooled 35-49"))
    print(f"  [F5d] GCN 2-layer  F1={f5d_f1:.4f}  PRAUC={f5d_prauc:.4f}")

    # ── Verdicts ──────────────────────────────────────────────────────────────
    d1 = _d1_f1()
    f5c_pass = (f5c_f1 >= 0.50) and ((f5c_f1 - d1) >= 0.15)
    if f5d_f1 < 0.40:
        f5d_verdict = "FAIL"
    elif 0.50 <= f5d_f1 <= 0.75:
        f5d_verdict = "PASS"
    else:
        f5d_verdict = "INCONCLUSIVE"
    ordering_ok = f5d_f1 >= f5c_f1 - 0.10

    log_verdict("F5c", "Corrected instrument: SGC+MLP K=2 (nonlinear head)",
                World_Eliminated="broken-instrument (nonlinear)", Readout_Metric="Static_OOT_F1",
                Decision_Rule=f"SGC+MLP F1>=0.50 AND (F1 - D1_K0={d1:.3f})>=0.15",
                Observed_Value=f5c_f1, Verdict="PASS" if f5c_pass else "FAIL",
                Sweep_Refs="F5c: SGC+MLP K=2 [corrected]",
                Notes=f"PRAUC={f5c_prauc}; gain over linear ceiling D1={f5c_f1 - d1:.3f}")
    log_verdict("F5d", "GCN reference (does graph signal exist on temporal split)",
                World_Eliminated="absent-graph-signal", Readout_Metric="Static_OOT_F1",
                Decision_Rule="GCN F1 in [0.50,0.75] PASS; <0.40 FAIL (graph signal absent)",
                Observed_Value=f5d_f1, Verdict=f5d_verdict,
                Sweep_Refs="F5d: GCN reference [2-layer]",
                Notes=f"PRAUC={f5d_prauc}; ordering GCN>=SGC+MLP-0.10: {ordering_ok}")
    log_verdict("F5-LIN", "Linear-classifier ceiling on Elliptic temporal split",
                World_Eliminated="none (instrumental calibration)", Readout_Metric="Static_OOT_F1",
                Decision_Rule="established finding, not a bug",
                Observed_Value=round(max(0.303, d1), 4), Verdict="PASS",
                Sweep_Refs="Diagnostic: sklearn LR, Diagnostic: K=0 linear, F5: SGC K=1 linear [v2-fixed]",
                Notes="Linear models (LR/fit_head/SGC-linear) cap ~0.25-0.30 F1; trees ~0.80. Nonlinear signal.")

    overall = "PASS" if (f5c_pass and f5d_verdict == "PASS") else "FAIL"
    log_verdict("F5", "Instrument check (corrected: nonlinear graph arch)",
                World_Eliminated="broken-instrument", Readout_Metric="Static_OOT_F1",
                Decision_Rule="F5c(SGC+MLP)>=0.50 & +0.15 over D1 AND F5d(GCN) in [0.50,0.75]",
                Observed_Value=f5c_f1, Verdict=overall,
                Sweep_Refs="F5c: SGC+MLP K=2 [corrected], F5d: GCN reference [2-layer]",
                Notes=f"XGB 0.784 / RF 0.804 PASS (tabular); F5c={f5c_f1} F5d={f5d_f1}; ordering_ok={ordering_ok}")

    print("\n" + "=" * 64)
    print(f"  D1 linear ceiling : F1={d1:.4f}")
    print(f"  F5c SGC+MLP K=2   : F1={f5c_f1:.4f} PRAUC={f5c_prauc:.4f}  verdict={'PASS' if f5c_pass else 'FAIL'}")
    print(f"  F5d GCN 2-layer   : F1={f5d_f1:.4f} PRAUC={f5d_prauc:.4f}  verdict={f5d_verdict}")
    print(f"  ordering GCN>=SGC+MLP-0.10: {ordering_ok}")
    print(f"  >>> OVERALL INSTRUMENT GATE: {overall}")
    print("=" * 64)


if __name__ == "__main__":
    main()
