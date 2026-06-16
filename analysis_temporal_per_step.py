"""
Generate per-timestep walk-forward F1 / PR-AUC for the LEARNED-phi temporal
models (SGC-LSTM and SGC-EMA), so they can be compared head-to-head against the
handcrafted-phi Temporal XGBoost rows already in walk_forward_timesteps.csv.

FREEZE-SAFE: this script imports and reuses the *frozen* evaluation primitives
from source/ (train_lstm_conditioned, train_ema_conditioned, _walk_forward_blocks,
_filtered_state, _find_best_f1_threshold). It does NOT modify any pipeline code.
It only replicates the thin per-step metric capture that the pooled evaluators
omit, and appends rows to results/walk_forward_timesteps.csv.

Usage:
    source venv/bin/activate
    python analysis_temporal_per_step.py --epochs 100
"""
import sys, os, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, average_precision_score

from config import Config, DEVICE, OUTPUT_DIR, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from evaluation.temporal_validation import (
    train_lstm_conditioned, train_ema_conditioned,
    _onestep_blocks, _temporal_state,
)
from evaluation.validation import _find_best_f1_threshold


def _per_step(dm, cfg, device, epochs, embed_dim, kind):
    """Run the frozen walk-forward loop, capturing per-tau F1 / PR-AUC.

    Mirrors walk_forward_{lstm,ema}_conditioned exactly (same blocks, same
    filtering state, same tau-1 threshold calibration) but records each step.
    """
    rows = []
    for tau in cfg.test_steps:
        train_block, calib_step, calib_state, infer_state = _onestep_blocks(dm.graphs, tau)
        if not train_block:
            continue

        g = dm.graphs[tau]
        m = g["labeled_mask"]
        if m.sum() == 0:
            continue
        yte = g["y"][m].numpy()
        if len(np.unique(yte)) < 2:
            continue

        if kind == "lstm":
            embedder, temporal, head = train_lstm_conditioned(
                dm, train_block, cfg, device, epochs=epochs, embed_dim=embed_dim
            )
            embedder.eval(); temporal.eval(); head.eval()
        else:  # ema
            embedder, temporal, head = train_ema_conditioned(
                dm, train_block, cfg, device, epochs=epochs, embed_dim=embed_dim
            )
            embedder.eval(); head.eval()

        # Threshold calibration on tau-1 (filtering state includes tau-1)
        threshold = 0.5
        if calib_step in dm.graphs:
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        h_cal = _temporal_state(embedder, temporal, calib_state, dm, device)
                        logits_cal = head(g_cal["prop"][m_cal].to(device), h_cal)
                        s_cal = torch.softmax(logits_cal, dim=1)[:, 1].cpu().numpy()
                    threshold = _find_best_f1_threshold(y_cal, s_cal)

        # Test on tau (one-step-ahead: state excludes tau)
        with torch.no_grad():
            h_tau = _temporal_state(embedder, temporal, infer_state, dm, device)
            logits_te = head(g["prop"][m].to(device), h_tau)
            s = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()

        y_pred = (s >= threshold).astype(int)
        step_f1 = float(f1_score(yte, y_pred, pos_label=1, zero_division=0))
        step_prauc = float(average_precision_score(yte, s))
        rows.append((tau, step_f1, step_prauc))
        print(f"  [{kind.upper()}] tau={tau:>2d}  F1={step_f1:.3f}  PR-AUC={step_prauc:.3f}", flush=True)
    return rows


def _append_rows(sweep_name, rows):
    csv_file = os.path.join(OUTPUT_DIR, "walk_forward_timesteps.csv")
    df = pd.DataFrame(
        {"Sweep": [sweep_name] * len(rows),
         "Timestep (tau)": [r[0] for r in rows],
         "F1": [r[1] for r in rows],
         "PR-AUC": [r[2] for r in rows]}
    )
    header = not os.path.exists(csv_file)
    df.to_csv(csv_file, mode="a", header=header, index=False)
    print(f"Appended {len(rows)} rows for '{sweep_name}' -> {csv_file}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100, help="WF epochs per tau (pipeline default 100)")
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--models", type=str, default="lstm,ema", help="comma list: lstm,ema")
    args = ap.parse_args()

    cfg = Config()  # deep-structural learned-phi representative: struct=True, K=2, late topo, MLP head
    set_global_seeds(cfg.seed)

    print("Loading raw dataset...", flush=True)
    df, df_edge, _, feature_cols = download_and_load_data()

    print("Building data module (SGC propagation + topology)... this is the slow part.", flush=True)
    t0 = time.time()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    print(f"Setup done in {time.time() - t0:.1f}s | sgc_input_dim={dm.sgc_input_dim}", flush=True)

    # LSTM must run on CPU under MPS (parity with sweep.py main())
    lstm_device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    if "ema" in wanted:
        print("\n=== SGC-EMA (learned phi, memoryless-ish baseline) ===", flush=True)
        t0 = time.time()
        rows = _per_step(dm, cfg, lstm_device, args.epochs, args.embed_dim, "ema")
        print(f"EMA walk-forward done in {time.time() - t0:.1f}s", flush=True)
        _append_rows("SGC-EMA Conditioned (learned phi)", rows)

    if "lstm" in wanted:
        print("\n=== SGC-LSTM (learned phi, deep structural) ===", flush=True)
        t0 = time.time()
        rows = _per_step(dm, cfg, lstm_device, args.epochs, args.embed_dim, "lstm")
        print(f"LSTM walk-forward done in {time.time() - t0:.1f}s", flush=True)
        _append_rows("SGC-LSTM Conditioned (learned phi)", rows)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
