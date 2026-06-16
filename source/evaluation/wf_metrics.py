"""Regime-stratified walk-forward metrics (F1 battery foundation).

Given per-τ predictions, produces:
  - per-τ rows (CSV-2 schema fields): Tau, N_labeled, N_illicit, N_licit,
    Low_Confidence, Regime, F1, PRAUC, Precision, Recall
  - an aggregate dict (CSV-1 schema fields): pooled + macro + regime-stratified
    (pre_shock τ≤42, shock τ=43, recovery τ≥44) F1 and PRAUC.

PRAUC (average precision) is the primary, threshold-free readout. F1 here uses a
fixed 0.5 cut on the supplied scores; callers that calibrate a threshold should
pass already-thresholded predictions via `y_pred` if they want F1 on that cut.
Any τ with < 10 illicit is flagged Low_Confidence.
"""
import numpy as np
from sklearn.metrics import (f1_score, average_precision_score,
                             precision_score, recall_score)

LOW_CONF_MIN_POS = 10


def regime_of(tau: int) -> str:
    if tau <= 42:
        return "pre_shock"
    if tau == 43:
        return "shock"
    return "recovery"


def _f1(y, yp):
    return float(f1_score(y, yp, pos_label=1, zero_division=0))


def _prauc(y, s):
    # average_precision needs at least one positive; undefined otherwise.
    if (y == 1).sum() == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _pool(records):
    if not records:
        return np.array([]), np.array([])
    y = np.concatenate([r["y_true"] for r in records])
    s = np.concatenate([r["scores"] for r in records])
    return y, s


def stratified_wf_metrics(records, threshold: float = 0.5):
    """records: list of {tau:int, y_true:np.ndarray, scores:np.ndarray}
    (optional per-record 'y_pred' overrides the 0.5 cut for F1).
    Returns (aggregate_dict, per_tau_rows)."""
    rows = []
    for r in sorted(records, key=lambda r: r["tau"]):
        tau, y, s = r["tau"], np.asarray(r["y_true"]), np.asarray(r["scores"])
        yp = r["y_pred"] if "y_pred" in r else (s >= threshold).astype(int)
        n_ill = int((y == 1).sum())
        n_lic = int((y == 0).sum())
        rows.append({
            "Tau": tau,
            "N_labeled": n_ill + n_lic,
            "N_illicit": n_ill,
            "N_licit": n_lic,
            "Low_Confidence": n_ill < LOW_CONF_MIN_POS,
            "Regime": regime_of(tau),
            "F1": round(_f1(y, yp), 4),
            "PRAUC": round(_prauc(y, s), 4),
            "Precision": round(float(precision_score(y, yp, pos_label=1, zero_division=0)), 4),
            "Recall": round(float(recall_score(y, yp, pos_label=1, zero_division=0)), 4),
        })

    def _agg_pool(recs):
        y, s = _pool(recs)
        if len(y) == 0:
            return float("nan"), float("nan")
        yp = (s >= threshold).astype(int)
        return round(_f1(y, yp), 4), round(_prauc(y, s), 4)

    pre = [r for r in records if regime_of(r["tau"]) == "pre_shock"]
    shock = [r for r in records if regime_of(r["tau"]) == "shock"]
    rec = [r for r in records if regime_of(r["tau"]) == "recovery"]

    pooled_f1, pooled_prauc = _agg_pool(records)
    pre_f1, pre_prauc = _agg_pool(pre)
    shock_f1, shock_prauc = _agg_pool(shock)
    rec_f1, rec_prauc = _agg_pool(rec)

    valid_f1 = [row["F1"] for row in rows]
    valid_prauc = [row["PRAUC"] for row in rows if not np.isnan(row["PRAUC"])]
    macro_f1 = round(float(np.mean(valid_f1)), 4) if valid_f1 else float("nan")
    macro_prauc = round(float(np.mean(valid_prauc)), 4) if valid_prauc else float("nan")

    agg = {
        "WF_Pooled_F1": pooled_f1, "WF_Pooled_PRAUC": pooled_prauc,
        "WF_Macro_F1": macro_f1, "WF_Macro_PRAUC": macro_prauc,
        "WF_Pre43_Pooled_F1": pre_f1, "WF_Pre43_PRAUC": pre_prauc,
        "WF_Shock_F1": shock_f1, "WF_Shock_PRAUC": shock_prauc,
        "WF_Recovery_Pooled_F1": rec_f1, "WF_Recovery_PRAUC": rec_prauc,
    }
    return agg, rows
