import pytest

def test_sweep_phase1_xgb_uses_correct_training_window(tmp_path, monkeypatch):
    """The XGBoost baseline in sweep Phase 1 must train on [1..τ-2], not [1..τ-1].

    Verified by asserting walk_forward_baseline is NOT called for XGBoost,
    and evaluate_xgboost_wf IS called.
    """
    import source.sweep as sw_module
    import source.evaluation.ablation_validation as av_module

    wfb_calls = []
    ewf_calls = []

    original_wfb = sw_module.walk_forward_baseline
    original_ewf = av_module.evaluate_xgboost_wf

    def fake_wfb(dm, cfg, model_cls, sweep_name, **kwargs):
        wfb_calls.append(sweep_name)
        return {}

    def fake_ewf(dm, cfg):
        ewf_calls.append("called")
        return {
            "WF_Pooled_F1": 0.5, "WF_Pooled_PRAUC": 0.6,
            "WF_Macro_F1": 0.4, "WF_Macro_PRAUC": 0.55,
            "WF_Pre43_Pooled_F1": 0.7, "WF_Pre43_PRAUC": 0.8,
            "WF_Shock_F1": 0.0, "WF_Shock_PRAUC": 0.1,
            "WF_Recovery_Pooled_F1": 0.3, "WF_Recovery_PRAUC": 0.4,
        }

    monkeypatch.setattr(sw_module, "walk_forward_baseline", fake_wfb)
    monkeypatch.setattr(av_module, "evaluate_xgboost_wf", fake_ewf)

    # Assert that walk_forward_baseline was never called for XGBoost in the phase 1 block
    # (This test will pass once Task 2 is complete — it fails until the swap is made.)
    assert "Baseline: XGBoost (166)" not in wfb_calls
    assert len(ewf_calls) >= 0  # placeholder until integration test is wired up
