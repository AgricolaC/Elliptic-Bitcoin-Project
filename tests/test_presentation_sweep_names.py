import ast, re

def _extract_string_literals(source: str, pattern: str) -> list[str]:
    """Find all string literals in source matching pattern."""
    return re.findall(pattern, source)

def test_presentation_xgb_sweep_names_match_ablation():
    """build_notebook.py must not reference XGBoost sweep names that the
    current pipeline never writes."""
    with open("source/reporting/build_notebook.py") as f:
        pres_src = f.read()
    with open("source/evaluation/ablation_validation.py") as f:
        abl_src = f.read()

    # Names the ablation actually writes
    abl_xgb_names = set(re.findall(r'"((?:Baseline|Ablation)[^"]*XGBoost[^"]*)"', abl_src))
    abl_xgb_names.update(re.findall(r"'((?:Baseline|Ablation)[^']*XGBoost[^']*)'", abl_src))

    # Names the presentation references
    pres_xgb_names = set(re.findall(r'"((?:F[0-9]+:|Baseline|Ablation)[^"]*(?:XGBoost|xgb)[^"]*)"', pres_src, re.IGNORECASE))
    pres_xgb_names.update(re.findall(r"'((?:F[0-9]+:|Baseline|Ablation)[^']*(?:XGBoost|xgb)[^']*)'", pres_src, re.IGNORECASE))

    stale = {n for n in pres_xgb_names if n not in abl_xgb_names and 'F1:' in n or 'F4:' in n}
    assert not stale, f"Presentation references sweep names not produced by ablation_validation.py: {stale}"
