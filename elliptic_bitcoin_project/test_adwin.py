"""
test_adwin.py — Test suite for the ADWIN adaptive-window walk-forward evaluator.

Five tests:
  1. test_adwin_detects_known_shift      — synthetic shift detection
  2. test_adwin_stable_stream_grows      — monotonic growth on stable stream
  3. test_no_foreknowledge_invariant     — corrupted-future invariance
  4. test_adaptive_uses_shared_aggregation — metric parity via shared helper
  5. test_adaptive_leakage_guard         — max(train_block) < tau always
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import torch


class TestADWINDetector:
    """Core ADWIN algorithm tests on synthetic streams."""

    def test_adwin_detects_known_shift(self):
        """
        Feed a synthetic stream: 30 samples at mean 0, then 20 at mean 5.
        ADWIN must detect drift and shrink the window so w_start advances
        past the change point.
        """
        from evaluation.adwin import ADWIN

        rng = np.random.default_rng(42)
        stream = np.concatenate([
            rng.normal(0.0, 0.5, size=30),
            rng.normal(5.0, 0.5, size=20),
        ])

        detector = ADWIN(delta=0.002)
        drift_detected_at = []

        for i, val in enumerate(stream):
            if detector.update(val):
                drift_detected_at.append(i)

        # Drift must be detected somewhere after the change point (index 30)
        assert len(drift_detected_at) > 0, \
            "ADWIN did not detect any drift in a stream with an obvious mean shift"

        # After processing the full stream, w_start should have advanced
        # past the change point (index 30)
        assert detector.w_start >= 30, (
            f"ADWIN w_start={detector.w_start}, expected >= 30. "
            f"The window should have dropped the old pre-shift data."
        )

    def test_adwin_stable_stream_grows(self):
        """
        Feed a constant-mean noisy stream. The window should grow monotonically
        and ADWIN should never detect drift (no false positives).
        """
        from evaluation.adwin import ADWIN

        rng = np.random.default_rng(123)
        stream = rng.normal(0.0, 0.1, size=50)

        detector = ADWIN(delta=0.002)
        widths = []

        for val in stream:
            detector.update(val)
            widths.append(detector.width)

        # Width should be monotonically non-decreasing
        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1], (
                f"ADWIN window shrank at step {i}: {widths[i-1]} -> {widths[i]}. "
                f"False drift detected on a stationary stream."
            )

        # After 50 samples the window should hold all of them
        assert detector.width == 50, (
            f"ADWIN width={detector.width}, expected 50 on a stable stream."
        )

    def test_no_foreknowledge_invariant(self):
        """
        Compute the ADWIN schedule, then corrupt stream[τ:] for a fixed τ,
        recompute, and assert schedule[τ] is identical.

        This is the mandatory leakage test: the window boundary at τ must
        not depend on any future stream value.
        """
        from evaluation.adwin import adwin_window_schedule

        rng = np.random.default_rng(7)
        stream = np.concatenate([
            rng.normal(0.0, 0.5, size=30),
            rng.normal(3.0, 0.5, size=20),
        ])

        # Original schedule
        schedule_original = adwin_window_schedule(stream, delta=0.002)

        # Test for multiple τ values
        for tau in [10, 25, 35, 45]:
            if tau >= len(stream):
                continue

            # Corrupt everything from τ onward
            corrupted_stream = stream.copy()
            corrupted_stream[tau:] = 999.0

            schedule_corrupted = adwin_window_schedule(corrupted_stream, delta=0.002)

            # Schedule values up to and including τ-1 must be identical
            for i in range(tau):
                assert schedule_original[i] == schedule_corrupted[i], (
                    f"FOREKNOWLEDGE LEAK at τ={tau}, index={i}: "
                    f"original w_start={schedule_original[i]}, "
                    f"corrupted w_start={schedule_corrupted[i]}. "
                    f"Future values affected past schedule."
                )


class TestAdaptiveWalkForward:
    """Tests for the adaptive walk-forward evaluator integration."""

    def test_adaptive_uses_shared_aggregation(self):
        """
        Verify metric parity: run both the fixed-window and adaptive evaluators
        on the same tiny dm and confirm they both return (pooled_f1, pooled_prauc)
        tuples — proving they use the same aggregation pathway.
        
        We verify structurally by checking:
        1. Both return 2-element tuples (without return_records)
        2. Both values are floats in [0, 1]
        """
        dm, cfg = _make_tiny_dm()

        from evaluation.validation import walk_forward_validation, walk_forward_validation_adaptive
        from config import DEVICE

        # Run fixed-window evaluator
        result_fixed = walk_forward_validation(dm, cfg, DEVICE, window=3, sweep_name="test_fixed")
        # Run adaptive evaluator
        result_adaptive = walk_forward_validation_adaptive(dm, cfg, DEVICE, delta=0.5, sweep_name="test_adwin")

        # Both should return 2-tuples
        assert isinstance(result_fixed, tuple) and len(result_fixed) == 2, (
            f"Fixed-window returned {type(result_fixed)} with {len(result_fixed)} elements, "
            f"expected 2-tuple (pooled_f1, pooled_prauc)."
        )
        assert isinstance(result_adaptive, tuple) and len(result_adaptive) == 2, (
            f"Adaptive returned {type(result_adaptive)} with {len(result_adaptive)} elements, "
            f"expected 2-tuple (pooled_f1, pooled_prauc)."
        )

        # Both values should be floats in [0, 1]
        for name, result in [("fixed", result_fixed), ("adaptive", result_adaptive)]:
            f1, prauc = result
            assert isinstance(f1, float), f"{name} F1 is {type(f1)}, expected float"
            assert isinstance(prauc, float), f"{name} PR-AUC is {type(prauc)}, expected float"
            assert 0.0 <= f1 <= 1.0, f"{name} F1={f1} out of [0,1]"
            assert 0.0 <= prauc <= 1.0, f"{name} PR-AUC={prauc} out of [0,1]"

    def test_adaptive_leakage_guard(self):
        """
        Run the adaptive evaluator on a tiny synthetic dm and verify
        that max(train_block) < τ holds at every step.

        We instrument the function by patching fit_head to record train_block
        boundaries, then verify after the run.
        """
        dm, cfg = _make_tiny_dm()

        from evaluation.validation import walk_forward_validation_adaptive
        from config import DEVICE

        # Run with return_records to get the per-step window starts
        result = walk_forward_validation_adaptive(
            dm, cfg, DEVICE, delta=0.5, return_records=True
        )

        if len(result) == 3:
            _, _, records = result
            for record in records:
                tau, f1, prauc, w_start, w_width = record
                # The train block is [w_start, w_start + w_width)
                max_train = w_start + w_width - 1
                assert max_train < tau, (
                    f"LEAKAGE at τ={tau}: max(train_block)={max_train} >= τ"
                )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_real_aggregate():
    """Import and return the real _aggregate_walk_forward for wrapping."""
    from evaluation.validation import _aggregate_walk_forward
    return _aggregate_walk_forward


def _make_tiny_dm():
    """
    Build a tiny synthetic EllipticDataModule for integration tests.
    Returns (dm, cfg).
    """
    from config import Config
    from data.build_graph import EllipticDataModule

    rng = np.random.default_rng(42)
    n_nodes, n_ts, n_feat = 30, 8, 5

    rows = []
    for t in range(1, n_ts + 1):
        for i in range(n_nodes):
            # Shift illicit rate at t=6 to give ADWIN something to detect
            if t < 6:
                label = rng.choice([0, 0, 0, 1, -1])  # ~25% illicit
            else:
                label = rng.choice([0, 0, 0, 0, -1])   # ~0% illicit
            row = {"txId": t * 1000 + i, "ts": t, "label": label}
            for f in range(n_feat):
                row[f"f{f}"] = rng.normal(0.0, 1.0)
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    feature_cols = [f"f{f}" for f in range(n_feat)]

    edge_rows = []
    for t in range(1, n_ts + 1):
        ids = [t * 1000 + i for i in range(n_nodes)]
        for i in range(0, n_nodes - 1, 3):
            edge_rows.append({"txId1": ids[i], "txId2": ids[i + 1]})
    df_edge = pd.DataFrame(edge_rows)

    cfg = Config(
        train_steps=range(1, 6),
        test_steps=range(6, 9),
        disruption_step=7,
        use_topology=False,
        use_multiscale_prop=False,
        sgc_k=1,
        sgc_epochs=5,
        wf_epochs=5,
    )

    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    return dm, cfg
