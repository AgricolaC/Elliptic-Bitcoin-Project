"""F2 — Sequence Control & Falsification Arena.

Evaluates the SGC-LSTM model under two strict conditions:
  1. Chronological (Normal): The training snapshots are fed to the BPTT sequence in order.
  2. Shuffled (Control): The training snapshots are randomly shuffled before being fed.

If Chronological ≈ Shuffled, it definitively proves the network has no exploitable sequence order,
and the sequence model is merely learning a broadcast bias.

Writes per-τ rows (CSV-2) and one aggregate row per model (CSV-1).
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(os.path.dirname(HERE))
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
import torch

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.validation import _calibrate_threshold
from evaluation.temporal_validation import (
    train_lstm_conditioned, _onestep_blocks, _temporal_state, _train_illicit_rate
)
from evaluation.wf_metrics import stratified_wf_metrics
from evaluation.falsification_log import log_verdict
from sweep import _make_result, _RESULT_KEYS
from execution.phases.f1_walk_forward import _write_csv1, _write_csv2, _migrate_csv2, SEED

def collect_temporal_lstm(dm, cfg, device, gir, epochs, embed_dim=128, shuffle_train=False):
    recs, extra = [], {}
    for tau in cfg.test_steps:
        tb, cal, calib_state, infer_state = _onestep_blocks(dm.graphs, tau)
        if not tb:
            continue
        g = dm.graphs[tau]; m = g["labeled_mask"]
        if m.sum() == 0 or len(np.unique(g["y"][m].numpy())) < 2:
            continue
            
        emb, temporal, head = train_lstm_conditioned(
            dm, tb, cfg, device, epochs=epochs, embed_dim=embed_dim, shuffle_train=shuffle_train
        )
        emb.eval(); temporal.eval(); head.eval()
        
        thr, fb = 0.5, False
        if cal in dm.graphs:
            g_cal = dm.graphs[cal]; m_cal = g_cal["labeled_mask"]
            yc = g_cal["y"][m_cal].numpy()
            if m_cal.sum() > 0 and len(np.unique(yc)) >= 2:
                with torch.no_grad():
                    h_cal = _temporal_state(emb, temporal, calib_state, dm, device)
                    sc = torch.softmax(head(g_cal["prop"][m_cal].to(device), h_cal), dim=1)[:, 1].cpu().numpy()
                thr, fb = _calibrate_threshold(yc, sc, gir)
        
        with torch.no_grad():
            h_tau = _temporal_state(emb, temporal, infer_state, dm, device)
            s = torch.softmax(head(g["prop"][m].to(device), h_tau), dim=1)[:, 1].cpu().numpy()
            
        yte = g["y"][m].numpy()
        recs.append({"tau": tau, "y_true": yte, "scores": s, "y_pred": (s >= thr).astype(int)})
        extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4),
                      "Calib_Fallback": bool(fb)}
        
        shuf_tag = "[Shuffled]" if shuffle_train else "[Chronological]"
        print(f"    {shuf_tag} τ={tau} done", flush=True)
        
    return recs, extra

def _run(name, collect_fn, feature_set, threshold_method="epsilon-fallback"):
    print(f"\\n=== {name} ===", flush=True)
    t0 = time.time()
    recs, extra = collect_fn()
    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(name, rows, extra)
    _write_csv1(name, agg, time.time() - t0, feature_set, threshold_method)
    print(f"  {name}: Pre43_PRAUC={agg['WF_Pre43_PRAUC']}  Recovery_PRAUC={agg['WF_Recovery_PRAUC']}  "
          f"Pooled_PRAUC={agg['WF_Pooled_PRAUC']}", flush=True)
    return agg

def _log_f2_verdict(a_lstm: dict, a_shuf: dict):
    """Compute F2 verdict, log it to falsification_log.csv, and return (verdict, sub)."""
    diff = a_lstm["WF_Pooled_PRAUC"] - a_shuf["WF_Pooled_PRAUC"]
    if abs(diff) <= 0.02:
        verdict = "PASS"
        sub = "Shuffled memory matches chronological memory; temporal order is uninformative."
    else:
        verdict = "FAIL"
        sub = f"Sequence matters! Diff={diff:.4f}"

    log_verdict(
        "F2",
        "Order-invariance check (shuffled vs chronological LSTM)",
        "World A (temporal drift detectable via sequence order)",
        "WF_Pooled_PRAUC_diff",
        "|LSTM_chron - LSTM_shuf| <= 0.02",
        round(abs(diff), 4),
        verdict,
        Notes=sub,
    )
    return verdict, sub


def run():
    set_global_seeds(SEED)
    print("Loading raw dataset for F2...", flush=True)
    df, df_edge, _, feature_cols = download_and_load_data()

    cfg = Config(train_steps=range(1, 27), val_steps=range(27, 35), test_steps=range(35, 50),
                 sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
                 use_graph_structural=False, use_directional_prop=False, seed=SEED)
                 
    print("Building data module (frozen preprocessing on train 1-26)...", flush=True)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    gir = _train_illicit_rate(dm, cfg)
    lstm_device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
    
    _migrate_csv2()

    # 1. Chronological LSTM
    a_lstm = _run("F2: SGC-LSTM Chronological",
                  lambda: collect_temporal_lstm(dm, cfg, lstm_device, gir, cfg.wf_epochs, shuffle_train=False),
                  "SGC K=2 + Chronological LSTM context")
                  
    # 2. Shuffled LSTM
    a_shuf = _run("F2: SGC-LSTM Shuffled",
                  lambda: collect_temporal_lstm(dm, cfg, lstm_device, gir, cfg.wf_epochs, shuffle_train=True),
                  "SGC K=2 + Shuffled LSTM context")

    verdict, sub = _log_f2_verdict(a_lstm, a_shuf)
    print(f"\n>>> F2 VERDICT: {verdict} — {sub}")

if __name__ == "__main__":
    run()
