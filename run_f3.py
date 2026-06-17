"""F3 — Temporal-shuffle control (MECHANISTIC EXHIBIT, not a gate).

F1 already settled the verdict: World B. The pre-shock PRAUC ladder is monotone
(Base XGB 0.936 ≥ Temporal XGB 0.935 > SGC+MLP static 0.866 > SGC-LSTM 0.761 >
SGC-EMA 0.711) — every added component costs PRAUC before any regime change
exists, and the LSTM loses to its own static substrate by ~0.10. F3 does NOT
re-open that; it only explains WHY, for the oral exam.

THE QUESTION: does the SGC-LSTM exploit temporal *order* at all? We rerun the
exact F1 SGC-LSTM walk-forward protocol but DESTROY chronology:
  • TRAINING:  the snapshot sequence fed to the LSTM is re-permuted every epoch
    (fresh draw from a per-τ seeded RNG). Per-snapshot feature content and its
    label supervision are untouched — only the order the LSTM sees changes. This
    forces order-invariance, so it tests whether order is *usable at all*, not
    whether one particular scramble is bad.
  • INFERENCE: the conditioning history that builds h_τ (and h_{τ-1} for calib)
    is also shuffled, so the test-time state is de-chronologized too. Shuffling
    only training would be a train/test mismatch, not a clean order ablation.
Everything else — frozen preprocessing, expanding window [1..τ-2], τ-1 calib
with ε-fallback, one-step-ahead state (τ excluded) — is identical to F1.

Compared against the UNSHUFFLED F1 SGC-LSTM (read from sweep_results.csv).
Hypothesis logged: "the recurrence does NOT exploit temporal order" (World-B
mechanism). Readout WF_Pre43_PRAUC, Δ = shuffled − unshuffled:
  |Δ| < 0.03            → PASS  : order-invariant → broadcast per-snapshot bias.
  Δ ≤ −0.05 (worse)     → FAIL  : it DOES use order (just badly) — alt mechanism.
  otherwise             → INCONCLUSIVE.
Either way the F1=FAIL/World-B verdict is unchanged. Stop after F3.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "source")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

import numpy as np
import pandas as pd
import torch

from config import Config, OUTPUT_DIR, DEVICE, set_global_seeds
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from models.temporal_head import SnapshotEmbedder, TemporalLSTM, LSTMConditionedHead
from models.classifier import build_loss
from evaluation.validation import _compute_class_weights, _calibrate_threshold
from evaluation.temporal_validation import _onestep_blocks, _train_illicit_rate
from evaluation.wf_metrics import stratified_wf_metrics
from evaluation.falsification_log import log_verdict
from sweep import _make_result, _RESULT_KEYS
from run_f1 import _write_csv2, _migrate_csv2, SEED, TS_CSV, SWEEP_CSV

UNSHUFFLED_SWEEP = "F1: SGC-LSTM WF [v2-fixed]"   # F1 baseline to compare against
SHUFFLED_SWEEP = "F3: SGC-LSTM Shuffled [v2-fixed]"


# ─────────────────────────── shuffle-aware LSTM ─────────────────────────────
def _shuffled_state(embedder, lstm, steps, dm, device, rng):
    """Embed ``steps`` and run the LSTM over them in a RANDOM order; return the
    final hidden state. Chronology destroyed, per-snapshot features unchanged."""
    avail = [t for t in steps if t in dm.graphs]
    order = [avail[i] for i in rng.permutation(len(avail))]
    embeddings = [embedder(dm.graphs[t]["prop"].to(device)) for t in order]
    return lstm(torch.stack(embeddings))[-1]


def train_lstm_shuffled(dm, train_steps, cfg, device, rng, epochs, embed_dim=32):
    """F1's train_lstm_conditioned with the training sequence re-permuted every
    epoch. hidden_states[i] is still supervised by the snapshot at sequence
    position i, so features↔labels stay paired; only the order the LSTM sees is
    randomized."""
    embedder = SnapshotEmbedder(dm.sgc_input_dim, embed_dim, cfg).to(device)
    lstm = TemporalLSTM(embed_dim, cfg.lstm_hidden).to(device)
    head = LSTMConditionedHead(dm.sgc_input_dim, cfg.lstm_hidden, cfg).to(device)
    params = list(embedder.parameters()) + list(lstm.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay)

    chrono = sorted([t for t in train_steps if t in dm.graphs])
    if not chrono:
        return embedder, lstm, head
    ytr_all = torch.cat([dm.graphs[t]["y"][dm.graphs[t]["labeled_mask"]] for t in chrono])
    cls_w = _compute_class_weights(ytr_all, device)
    loss_fn = build_loss(cfg, cls_w)

    embedder.train(); lstm.train(); head.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        order = [chrono[i] for i in rng.permutation(len(chrono))]   # SHUFFLE
        embeddings = [embedder(dm.graphs[t]["prop"].to(device)) for t in order]
        hidden = lstm(torch.stack(embeddings))
        total = 0.0
        for i, t in enumerate(order):
            g = dm.graphs[t]; m = g["labeled_mask"]
            if m.sum() > 0:
                total += loss_fn(head(g["prop"][m].to(device), hidden[i]), g["y"][m].to(device))
        if total > 0:
            total.backward(); opt.step()
    return embedder, lstm, head


def collect_lstm_shuffled(dm, cfg, device, gir, epochs, embed_dim=32):
    recs, extra = [], {}
    for tau in cfg.test_steps:
        tb, cal, calib_state, infer_state = _onestep_blocks(dm.graphs, tau)
        if not tb:
            continue
        g = dm.graphs[tau]; m = g["labeled_mask"]
        if m.sum() == 0 or len(np.unique(g["y"][m].numpy())) < 2:
            continue
        rng = np.random.RandomState(SEED + tau)   # per-τ, reproducible, order-independent
        emb, lstm, head = train_lstm_shuffled(dm, tb, cfg, device, rng, epochs, embed_dim)
        emb.eval(); lstm.eval(); head.eval()

        thr, fb = 0.5, False
        if cal in dm.graphs:
            g_cal = dm.graphs[cal]; m_cal = g_cal["labeled_mask"]
            yc = g_cal["y"][m_cal].numpy()
            if m_cal.sum() > 0 and len(np.unique(yc)) >= 2:
                with torch.no_grad():
                    h_cal = _shuffled_state(emb, lstm, calib_state, dm, device, rng)
                    sc = torch.softmax(head(g_cal["prop"][m_cal].to(device), h_cal), dim=1)[:, 1].cpu().numpy()
                thr, fb = _calibrate_threshold(yc, sc, gir)
        with torch.no_grad():
            h_tau = _shuffled_state(emb, lstm, infer_state, dm, device, rng)
            s = torch.softmax(head(g["prop"][m].to(device), h_tau), dim=1)[:, 1].cpu().numpy()
        yte = g["y"][m].numpy()
        recs.append({"tau": tau, "y_true": yte, "scores": s, "y_pred": (s >= thr).astype(int)})
        extra[tau] = {"Train_Window_Size": len(tb), "Calib_Threshold": round(float(thr), 4),
                      "Calib_Fallback": bool(fb)}
        print(f"    [shuffle] τ={tau} done", flush=True)
    return recs, extra


# ─────────────────────────── CSV-1 writer (F3 notes) ────────────────────────
def _write_csv1_f3(sweep, agg, wf_time, feature_set):
    df_new = pd.DataFrame([_make_result(
        seed=SEED, variation="Base", sweep=sweep,
        static_time="N/A", static_mem="N/A", static_f1="N/A", static_prauc="N/A",
        wf_time=round(wf_time, 2), wf_mem="N/A",
        wf_f1=agg["WF_Macro_F1"], wf_prauc=agg["WF_Macro_PRAUC"],
        wf_pooled_f1=agg["WF_Pooled_F1"], wf_pooled_prauc=agg["WF_Pooled_PRAUC"],
        wf_pre43_pooled_f1=agg["WF_Pre43_Pooled_F1"], wf_pre43_prauc=agg["WF_Pre43_PRAUC"],
        wf_shock_f1=agg["WF_Shock_F1"], wf_shock_prauc=agg["WF_Shock_PRAUC"],
        wf_recovery_pooled_f1=agg["WF_Recovery_Pooled_F1"], wf_recovery_prauc=agg["WF_Recovery_PRAUC"],
        feature_set=feature_set, threshold_method="epsilon-fallback", selfcond_bug="fixed",
        notes="F3 temporal-shuffle control: per-epoch random snapshot order + shuffled inference state",
    )], columns=list(_RESULT_KEYS))
    df = pd.concat([pd.read_csv(SWEEP_CSV, keep_default_na=False), df_new], ignore_index=True)
    df.to_csv(SWEEP_CSV, index=False)


def _read_unshuffled_pre43():
    """Read the F1 (unshuffled) SGC-LSTM WF_Pre43_PRAUC from sweep_results.csv."""
    df = pd.read_csv(SWEEP_CSV, keep_default_na=False)
    hit = df[df["Sweep"] == UNSHUFFLED_SWEEP]
    if len(hit) == 0:
        raise SystemExit(f"Cannot find F1 baseline row '{UNSHUFFLED_SWEEP}' — run run_f1.py first.")
    return float(hit.iloc[-1]["WF_Pre43_PRAUC"])


def main():
    set_global_seeds(SEED)
    print("Loading raw dataset...", flush=True)
    df, df_edge, _, feature_cols = download_and_load_data()
    assert "ts" not in feature_cols

    cfg = Config(train_steps=range(1, 27), val_steps=range(27, 35), test_steps=range(35, 50),
                 sgc_k=2, use_multiscale_prop=True, use_mlp_head=True,
                 use_graph_structural=False, use_directional_prop=False, seed=SEED)
    print("Building data module (frozen preprocessing on train 1-26)...", flush=True)
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    gir = _train_illicit_rate(dm, cfg)
    lstm_device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
    _migrate_csv2()

    unshuffled = _read_unshuffled_pre43()

    print(f"\n=== {SHUFFLED_SWEEP} ===", flush=True)
    t0 = time.time()
    recs, extra = collect_lstm_shuffled(dm, cfg, lstm_device, gir, cfg.wf_epochs)
    agg, rows = stratified_wf_metrics(recs)
    _write_csv2(SHUFFLED_SWEEP, rows, extra)
    _write_csv1_f3(SHUFFLED_SWEEP, agg, time.time() - t0, "SGC prop + LSTM context (order shuffled)")

    shuffled = agg["WF_Pre43_PRAUC"]
    delta = round(float(shuffled) - float(unshuffled), 4)
    if abs(delta) < 0.03:
        verdict, sub = "PASS", ("order-invariant: shuffling chronology does not change pre-shock "
                                "PRAUC → the LSTM acts as a per-snapshot broadcast bias, not a sequence model")
    elif delta <= -0.05:
        verdict, sub = "FAIL", ("shuffling clearly hurts → the LSTM DOES exploit temporal order "
                                "(but still underperforms static SGC+MLP) — alternative World-B mechanism")
    else:
        verdict, sub = "INCONCLUSIVE", f"weak/ambiguous order sensitivity (Δ={delta})"

    log_verdict(
        "F3", "Temporal-shuffle control (mechanistic exhibit, not a gate)",
        World_Eliminated="recurrence-uses-temporal-order (tests World-B mechanism, not the verdict)",
        Readout_Metric="WF_Pre43_PRAUC",
        Decision_Rule="|shuffled − unshuffled| < 0.03 → PASS (order-invariant); Δ ≤ −0.05 → FAIL (uses order)",
        Observed_Value=delta, Verdict=verdict,
        Sweep_Refs=f"{SHUFFLED_SWEEP}, {UNSHUFFLED_SWEEP}",
        Notes=(f"{sub}. Pre43 PRAUC: unshuffled={unshuffled}, shuffled={shuffled}, Δ={delta}. "
               f"Does NOT change F1=FAIL/World-B. Exhibit for oral exam."))

    print("\n" + "=" * 70)
    print("F3 TEMPORAL-SHUFFLE CONTROL (pre-shock PRAUC):")
    print(f"  SGC-LSTM unshuffled (F1) : {unshuffled}")
    print(f"  SGC-LSTM shuffled  (F3)  : {shuffled}")
    print(f"  Δ (shuffled − unshuffled): {delta}")
    print(f"  >>> F3 VERDICT: {verdict} — {sub}")
    print("  (F1=FAIL/World-B stands; F3 only explains the mechanism.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
