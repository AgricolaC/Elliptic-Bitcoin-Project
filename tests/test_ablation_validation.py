import pytest
from unittest.mock import MagicMock, patch

def test_evaluate_xgboost_wf_returns_agg_not_result_dict():
    """evaluate_xgboost_wf must return an agg dict, not a _make_result dict."""
    # We patch the expensive internals; we only care about the return type/keys.
    from evaluation.ablation_validation import evaluate_xgboost_wf

    # Minimal agg-shaped return
    fake_agg = {
        "WF_Pooled_F1": 0.5, "WF_Pooled_PRAUC": 0.6,
        "WF_Macro_F1": 0.4, "WF_Macro_PRAUC": 0.55,
        "WF_Pre43_Pooled_F1": 0.7, "WF_Pre43_PRAUC": 0.8,
        "WF_Shock_F1": 0.0, "WF_Shock_PRAUC": 0.1,
        "WF_Recovery_Pooled_F1": 0.3, "WF_Recovery_PRAUC": 0.4,
    }

    with patch("evaluation.ablation_validation.stratified_wf_metrics", return_value=(fake_agg, [])), \
         patch("evaluation.ablation_validation._write_csv2"), \
         patch("evaluation.ablation_validation._tab_block", return_value=(None, None)), \
         patch("evaluation.ablation_validation._walk_forward_blocks", return_value=([], 34)), \
         patch("evaluation.ablation_validation.profile_resources") as mock_ctx:
        mock_ctx.return_value.__enter__ = MagicMock(return_value={})
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        dm = MagicMock()
        dm.graphs = {}
        cfg = MagicMock()
        cfg.test_steps = []

        result = evaluate_xgboost_wf(dm, cfg)

    # Must be the agg dict, not a _make_result dict
    assert "WF_Pooled_F1" in result
    assert "Sweep" not in result  # _make_result dicts have 'Sweep'
    assert "Static_OOT_Pooled_F1" not in result
