"""
Adversarial test suite for the GL-TVD remediation plan.
Follows axiom-falsify: every test is written to FAIL before the fix exists,
then verified to PASS after the fix is applied.

Five fix targets:
  W1  — mixed feature scaling (build_graph.py)
  W3  — NaN bypass in temporal leakage guard (load_dataset.py)
  W2  — stacking meta distribution shift (stacking.py)
  W6  — sgc_input_dim not encapsulated (build_graph.py)
  W7/W8 — walk-forward plot overwrite + static class weights (validation.py)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# W1 — Feature scaling must cover topology + recon columns
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureScaling:
    """
    W1: After setup(), every column in every graph's x tensor must be
    approximately zero-mean and unit-variance *on the training split*.
    PageRank (~1e-4) and clustering (~0..1) were previously unscaled.
    """

    def _make_minimal_dm(self, use_topology: bool, use_recon: bool):
        """Build a tiny synthetic EllipticDataModule for scaling checks."""
        from config import Config
        from data.build_graph import EllipticDataModule

        n_nodes, n_ts, n_feat = 60, 6, 10

        rng = np.random.default_rng(0)
        rows = []
        for t in range(1, n_ts + 1):
            for i in range(n_nodes):
                row = {"txId": t * 1000 + i, "ts": t, "label": rng.choice([0, 1, -1])}
                for f in range(n_feat):
                    row[f"f{f}"] = rng.normal(100.0, 50.0)  # large mean/std on purpose
                rows.append(row)

        df = pd.DataFrame(rows)
        feature_cols = [f"f{f}" for f in range(n_feat)]

        # Build a sparse edge list within each timestep
        edge_rows = []
        for t in range(1, n_ts + 1):
            ids = [t * 1000 + i for i in range(n_nodes)]
            for i in range(0, n_nodes - 1, 3):
                edge_rows.append({"txId1": ids[i], "txId2": ids[i + 1]})
        df_edge = pd.DataFrame(edge_rows)

        cfg = Config(
            train_steps=range(1, 5),
            test_steps=range(5, 7),
            use_topology=use_topology,
            use_recon_error=use_recon,
            svd_components=3,
        )
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()
        return dm, cfg

    def test_base_features_are_scaled(self):
        """Base 166-dim block must be ~N(0,1) on the training split."""
        dm, cfg = self._make_minimal_dm(use_topology=False, use_recon=False)
        Xs = np.concatenate([dm.graphs[t]["x"].numpy() for t in cfg.train_steps])
        col_means = np.abs(Xs.mean(axis=0))
        col_stds  = Xs.std(axis=0)
        assert col_means.max() < 0.1,  f"Base feature mean off: max={col_means.max():.4f}"
        assert col_stds.min()  > 0.5,  f"Base feature std off:  min={col_stds.min():.4f}"
        assert col_stds.max()  < 2.0,  f"Base feature std off:  max={col_stds.max():.4f}"

    def test_topology_columns_are_scaled_W1(self):
        """
        W1 REGRESSION: After fix, the topology columns (PageRank, clustering)
        appended at index n_base: must also be ~N(0,1) on the training split.
        PageRank raw values are O(1e-4); unscaled they dominate at the wrong scale.
        """
        dm, cfg = self._make_minimal_dm(use_topology=True, use_recon=False)
        Xs = np.concatenate([dm.graphs[t]["x"].numpy() for t in cfg.train_steps])
        n_base = 10   # synthetic feature count
        topo_cols = Xs[:, n_base: n_base + 2]   # pagerank + clustering

        topo_means = np.abs(topo_cols.mean(axis=0))
        topo_stds  = topo_cols.std(axis=0)
        assert topo_means.max() < 0.5, (
            f"W1: Topology columns NOT scaled — mean={topo_means}. "
            f"Fix: apply a second StandardScaler pass after topology injection."
        )
        assert topo_stds.min() > 0.1, (
            f"W1: Topology columns appear constant — std={topo_stds}."
        )

    def test_recon_error_column_is_scaled_W1(self):
        """
        W1 REGRESSION: Reconstruction error column must be ~N(0,1) on train split.
        """
        dm, cfg = self._make_minimal_dm(use_topology=True, use_recon=True)
        Xs = np.concatenate([dm.graphs[t]["x"].numpy() for t in cfg.train_steps])
        recon_col = Xs[:, -1]  # last column is recon error
        mean_abs = abs(recon_col.mean())
        std_val  = recon_col.std()
        assert mean_abs < 0.5, (
            f"W1: Recon error column NOT scaled — mean={mean_abs:.4f}. "
            f"Fix: include recon error in second scaler pass."
        )
        assert std_val > 0.1, f"W1: Recon error appears constant — std={std_val:.4f}."

    def test_second_scaler_fitted_on_train_only_no_leakage(self):
        """
        LEAKAGE GUARD: The second-pass scaler must be fitted on train_steps only.
        Verify by checking that test-step stats are NOT the same as train-step stats
        when the test distribution intentionally differs.
        """
        from config import Config
        from data.build_graph import EllipticDataModule

        rng = np.random.default_rng(1)
        n_feat = 10
        rows = []
        for t in range(1, 7):
            mean = 0.0 if t < 5 else 100.0   # deliberate distribution shift at test time
            for i in range(40):
                row = {"txId": t * 1000 + i, "ts": t, "label": rng.choice([0, 1, -1])}
                for f in range(n_feat):
                    row[f"f{f}"] = rng.normal(mean, 1.0)
                rows.append(row)

        df = pd.DataFrame(rows)
        feature_cols = [f"f{f}" for f in range(n_feat)]
        df_edge = pd.DataFrame(columns=["txId1", "txId2"])

        cfg = Config(
            train_steps=range(1, 5),
            test_steps=range(5, 7),
            use_topology=False,
            use_recon_error=False,
            svd_components=2,
        )
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()

        # Test nodes should have large positive means (they come from N(100,1))
        # but standardized against train (N(0,1)). So test means should be ~100 in
        # standardized space — very far from zero.
        Xte = np.concatenate([dm.graphs[t]["x"].numpy() for t in cfg.test_steps])
        test_mean = Xte.mean()
        assert test_mean > 10.0, (
            f"LEAKAGE GUARD: Second scaler must use train stats. "
            f"Test mean after transform = {test_mean:.2f} (expected >> 0 due to shift). "
            f"If ~0, scaler was (re)fitted on test data."
        )


# ─────────────────────────────────────────────────────────────────────────────
# W3 — NaN bypass in temporal leakage guard
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalLeakageGuard:
    """
    W3: The guard must reject orphan edge txIds (not in df) rather than
    silently treating NaN != NaN as False and passing.
    """

    def _make_df_and_edge(self):
        df = pd.DataFrame({
            "txId": [1001, 1002, 1003],
            "ts":   [1,    1,    1   ],
            "label":[0,    1,    -1  ],
        })
        # Normal edge within t=1 (valid)
        df_edge = pd.DataFrame({
            "txId1": [1001],
            "txId2": [1002],
        })
        return df, df_edge

    def test_valid_edges_pass_guard(self):
        """Sanity: a clean dataset passes the leakage guard without error."""
        from data.load_dataset import _validate_temporal_edges
        df, df_edge = self._make_df_and_edge()
        # Should not raise
        _validate_temporal_edges(df, df_edge)

    def test_orphan_edge_raises_W3(self):
        """
        W3 REGRESSION: An edge containing a txId absent from df must raise,
        not silently pass due to NaN == NaN being False.
        """
        from data.load_dataset import _validate_temporal_edges
        df, df_edge = self._make_df_and_edge()
        # 9999 is not in df — old code would produce NaN, skip it, pass the guard
        df_edge_orphan = pd.DataFrame({"txId1": [9999], "txId2": [1001]})
        with pytest.raises(AssertionError, match="orphan"):
            _validate_temporal_edges(df, df_edge_orphan)

    def test_cross_temporal_edge_raises(self):
        """An edge spanning two distinct timesteps must always raise."""
        from data.load_dataset import _validate_temporal_edges
        df = pd.DataFrame({
            "txId": [1001, 2001],
            "ts":   [1,    2   ],
            "label":[0,    1   ],
        })
        df_edge_cross = pd.DataFrame({"txId1": [1001], "txId2": [2001]})
        with pytest.raises(AssertionError, match="temporal"):
            _validate_temporal_edges(df, df_edge_cross)


# ─────────────────────────────────────────────────────────────────────────────
# W2 — Stacking meta distribution shift
# ─────────────────────────────────────────────────────────────────────────────

class TestStackingMetaDistribution:
    """
    W2: The base models generating meta-training features and meta-test features
    must be trained on the same set of timesteps (early window only).
    """

    def test_base_model_window_consistency_W2(self):
        """
        W2 REGRESSION: Capture which fit_steps are passed to base_predictions
        for the meta-train call and the meta-test call.
        Both must use early steps only.
        """
        from models.stacking import stacking_meta_classifier

        # We don't need the real dataset — just check the call contract.
        # Patch base_predictions to record what fit_steps it is called with.
        recorded_fit_steps = []

        dm_mock = MagicMock()
        dm_mock.sgc_input_dim = 10

        from config import Config
        cfg = Config(
            train_steps=range(1, 11),
            test_steps=range(11, 16),
        )
        early = [s for s in cfg.train_steps if s <= 5]

        with patch("models.stacking._base_predictions") as mock_bp:
            # Return dummy arrays so the LR can fit
            n_meta = 20
            n_test = 10
            mock_bp.side_effect = [
                (np.random.rand(n_meta, 3), np.random.randint(0, 2, n_meta)),
                (np.random.rand(n_test, 3), np.random.randint(0, 2, n_test)),
            ]
            try:
                stacking_meta_classifier(dm_mock, cfg)
            except Exception:
                pass  # we only care about the call args

            calls = mock_bp.call_args_list
            assert len(calls) == 2, f"Expected 2 calls to _base_predictions, got {len(calls)}"

            # Both calls must pass early as fit_steps
            meta_train_fit_steps = list(calls[0][0][0])  # first positional arg of first call
            meta_test_fit_steps  = list(calls[1][0][0])  # first positional arg of second call

            assert meta_train_fit_steps == early, (
                f"W2: Meta-train base models fitted on {meta_train_fit_steps}, "
                f"expected early={early}."
            )
            assert meta_test_fit_steps == early, (
                f"W2: Meta-test base models fitted on {meta_test_fit_steps}, "
                f"expected early={early}. Old code used cfg.train_steps here — "
                f"that causes distribution shift."
            )


# ─────────────────────────────────────────────────────────────────────────────
# W6 — sgc_input_dim encapsulation
# ─────────────────────────────────────────────────────────────────────────────

class TestSGCInputDimEncapsulation:
    """
    W6: EllipticDataModule.setup() must set sgc_input_dim internally.
    Callers must never need to assign it externally.
    """

    def test_sgc_input_dim_set_after_setup_W6(self):
        """
        W6 REGRESSION: After setup(), dm.sgc_input_dim must be accessible
        without any external assignment.
        """
        from config import Config
        from data.build_graph import EllipticDataModule

        rng = np.random.default_rng(0)
        n_feat = 8
        rows = []
        for t in range(1, 5):
            for i in range(20):
                row = {"txId": t * 100 + i, "ts": t, "label": rng.choice([0, 1, -1])}
                for f in range(n_feat):
                    row[f"f{f}"] = rng.normal()
                rows.append(row)

        df = pd.DataFrame(rows)
        feature_cols = [f"f{f}" for f in range(n_feat)]
        df_edge = pd.DataFrame(columns=["txId1", "txId2"])

        cfg = Config(
            train_steps=range(1, 4),
            test_steps=range(4, 5),
            use_topology=False,
            use_recon_error=False,
            use_multiscale_prop=True,
            sgc_k=2,
        )
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()   # <-- should propagate internally and set sgc_input_dim

        # Must not raise AttributeError
        assert hasattr(dm, "sgc_input_dim"), (
            "W6: dm.sgc_input_dim not set by setup(). "
            "Callers cannot be required to do: dm.sgc_input_dim = dm.graphs[1]['prop'].shape[1]"
        )
        expected_dim = n_feat * (cfg.sgc_k + 1)   # multiscale
        assert dm.sgc_input_dim == expected_dim, (
            f"W6: sgc_input_dim={dm.sgc_input_dim}, expected {expected_dim} "
            f"(n_feat={n_feat}, K={cfg.sgc_k}, multiscale=True)"
        )

    def test_prop_key_present_after_setup_W6(self):
        """Every graph dict must have a 'prop' key after setup() returns."""
        from config import Config
        from data.build_graph import EllipticDataModule

        rng = np.random.default_rng(1)
        n_feat = 5
        rows = []
        for t in range(1, 4):
            for i in range(15):
                row = {"txId": t * 100 + i, "ts": t, "label": 0}
                for f in range(n_feat):
                    row[f"f{f}"] = rng.normal()
                rows.append(row)

        df = pd.DataFrame(rows)
        feature_cols = [f"f{f}" for f in range(n_feat)]
        df_edge = pd.DataFrame(columns=["txId1", "txId2"])

        cfg = Config(
            train_steps=range(1, 3),
            test_steps=range(3, 4),
            use_topology=False,
            use_recon_error=False,
            sgc_k=1,
        )
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()

        for t in range(1, 4):
            assert "prop" in dm.graphs[t], (
                f"W6: dm.graphs[{t}] missing 'prop' key after setup(). "
                f"Propagation must happen inside setup(), not by the caller."
            )


# ─────────────────────────────────────────────────────────────────────────────
# W7 — Plot filename must not be static (sweep-name parameterized)
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardPlotNaming:
    """W7: walk_forward_validation must accept a sweep_name and embed it in the filename."""

    def test_plot_filename_contains_sweep_name_W7(self):
        """
        W7 REGRESSION: walk_forward_validation must write to a file whose name
        includes the sweep_name argument, preventing silent overwrite across sweeps.
        """
        import inspect
        from evaluation.validation import walk_forward_validation
        sig = inspect.signature(walk_forward_validation)
        assert "sweep_name" in sig.parameters, (
            "W7: walk_forward_validation missing 'sweep_name' parameter. "
            "All sweeps currently write to the same walk_forward_drift.png."
        )


# ─────────────────────────────────────────────────────────────────────────────
# W8 — Dynamic class weights inside walk-forward loop
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicClassWeights:
    """
    W8: Class weights in the walk-forward loop must be computed from the
    expanding training window [1..tau-1], not from the fixed static train set.
    """

    def test_walk_forward_does_not_accept_external_cls_w_W8(self):
        """
        W8 REGRESSION: walk_forward_validation must NOT accept a pre-computed
        cls_w argument. It must compute class weights internally per tau.
        If it still accepts cls_w externally, the caller can pass stale weights.
        """
        import inspect
        from evaluation.validation import walk_forward_validation
        sig = inspect.signature(walk_forward_validation)
        assert "cls_w" not in sig.parameters, (
            "W8: walk_forward_validation still accepts cls_w externally. "
            "Class weights must be computed inside the tau loop from the "
            "expanding window to avoid stale-weight miscalibration."
        )


# ─────────────────────────────────────────────────────────────────────────────
# W5 — Sweep result dict keys must be standardized
# ─────────────────────────────────────────────────────────────────────────────

class TestSweepResultKeyStandardization:
    """W5: All result dicts returned by sweeps must use identical metric key names."""

    REQUIRED_KEYS = {
        "Sweep",
        "Static OOT F1",
        "Static OOT PR-AUC",
        "Walk-Forward Mean F1",
        "Walk-Forward Mean PR-AUC",
    }

    def _check_keys(self, result: dict, source: str):
        missing = self.REQUIRED_KEYS - set(result.keys())
        extra   = set(result.keys()) - self.REQUIRED_KEYS
        assert not missing, f"W5: {source} result missing keys: {missing}"
        assert not extra,   f"W5: {source} result has unexpected keys: {extra}"

    def test_run_single_sweep_key_schema(self):
        """run_single_sweep must return the standardized key schema."""
        from run_sweeps import run_single_sweep
        # We mock the expensive internals; we only need to inspect the returned dict shape.
        with patch("run_sweeps.EllipticDataModule") as MockDM, \
             patch("run_sweeps.stack_prop") as mock_sp, \
             patch("run_sweeps.fit_head") as mock_fh, \
             patch("run_sweeps.walk_forward_validation") as mock_wf:

            mock_dm = MagicMock()
            mock_dm.graphs = {t: {"prop": torch.zeros(5, 10), "y": torch.zeros(5).long(),
                                  "labeled_mask": torch.ones(5).bool()} for t in range(1, 50)}
            mock_dm.sgc_input_dim = 10
            MockDM.return_value = mock_dm

            mock_sp.return_value = (torch.zeros(20, 10), torch.zeros(20).long())
            mock_model = MagicMock()
            mock_model.return_value = torch.zeros(20, 2)
            mock_fh.return_value = mock_model
            mock_wf.return_value = (0.7, 0.8)

            from config import Config
            cfg = Config(train_steps=range(1, 35), test_steps=range(35, 50))
            result = run_single_sweep("Test Sweep", cfg, MagicMock(), MagicMock(), [])
            self._check_keys(result, "run_single_sweep")
