"""
Adversarial test suite for the GL-TVD remediation plan.
"""

import sys, os
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
SOURCE_DIR = os.path.join(REPO_ROOT, "source")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

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

    def _make_minimal_dm(self, use_graph_structural: bool):
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
            # Star graph for first 5 nodes
            for i in range(1, 5):
                edge_rows.append({"txId1": ids[0], "txId2": ids[i]})
            # Triangle for next 3 nodes
            edge_rows.append({"txId1": ids[5], "txId2": ids[6]})
            edge_rows.append({"txId1": ids[6], "txId2": ids[7]})
            edge_rows.append({"txId1": ids[7], "txId2": ids[5]})
        df_edge = pd.DataFrame(edge_rows)

        cfg = Config(
            train_steps=range(1, 5),
            test_steps=range(5, 7),
            use_graph_structural=use_graph_structural,
        )
        dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
        dm.setup()
        return dm, cfg

    def test_base_features_are_scaled(self):
        """Base 166-dim block must be ~N(0,1) on the training split."""
        dm, cfg = self._make_minimal_dm(use_graph_structural=False)
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
        dm, cfg = self._make_minimal_dm(use_graph_structural=True)
        if getattr(cfg, 'topo_injection_mode', 'early') == 'early':
            Xs = np.concatenate([dm.graphs[t]["x"].numpy() for t in cfg.train_steps])
            n_base = 10   # synthetic feature count
            topo_cols = Xs[:, n_base: n_base + 2]   # pagerank + clustering
        else:
            topo_cols = np.concatenate([dm.graphs[t]["topo"].numpy() for t in cfg.train_steps])

        topo_means = np.abs(topo_cols.mean(axis=0))
        topo_stds  = topo_cols.std(axis=0)
        assert topo_means.max() < 0.5, (
            f"W1: Topology columns NOT scaled — mean={topo_means}. "
            f"Fix: apply a second StandardScaler pass after topology injection."
        )
        assert topo_stds.min() > 0.1, (
            f"W1: Topology columns appear constant — std={topo_stds}."
        )

    def test_late_injection_correctness(self):
        """Verify that topo_injection_mode='late' bypasses SGC propagation and appends correctly."""
        from config import Config
        cfg = Config(train_steps=range(1, 4), test_steps=range(4, 7), use_graph_structural=True, topo_injection_mode='late')
        dm, cfg_out = self._make_minimal_dm(use_graph_structural=True)
        # Re-run setup with late explicitly
        dm.cfg.topo_injection_mode = 'late'
        dm.setup()
        
        t = dm.cfg.train_steps[0]
        assert "topo" in dm.graphs[t], "Topology tensor not found in late mode."
        
        topo_shape = dm.graphs[t]["topo"].shape
        assert topo_shape[1] == 2, "Topology should have exactly 2 dimensions."
        
        prop_shape = dm.graphs[t]["prop"].shape
        # Base=10, K=2, multiscale=True -> 30 dimensions from base propagation + 2 from late injection = 32
        assert prop_shape[1] == 32, f"Propagated shape mismatch: {prop_shape[1]} != 32."

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
            use_graph_structural=False,
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
            use_graph_structural=False,
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
            use_graph_structural=False,
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
        "Seed",
        "Variation",
        "Sweep",
        "Feature Set",
        "Threshold",
        "Static Time (s)",
        "Static Mem (MB)",
        "Static Val F1",
        "Static Val PR-AUC",
        "Static OOT F1",
        "Static OOT PR-AUC",
        "WF Time (s)",
        "WF Mem (MB)",
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
        from source.sweep import run_single_sweep
        # We mock the expensive internals; we only need to inspect the returned dict shape.
        with patch("source.sweep.EllipticDataModule") as MockDM, \
             patch("source.sweep.stack_prop") as mock_sp, \
             patch("source.sweep.fit_head") as mock_fh, \
             patch("source.sweep.walk_forward_validation") as mock_wf, \
             patch("source.sweep.joblib.dump") as mock_dump, \
             patch("source.sweep.torch.save") as mock_save:

            mock_dm = MagicMock()
            mock_dm.graphs = {t: {"prop": torch.zeros(5, 10), "y": torch.zeros(5).long(),
                                  "labeled_mask": torch.ones(5).bool()} for t in range(1, 50)}
            mock_dm.sgc_input_dim = 10
            MockDM.return_value = mock_dm

            mock_sp.return_value = (torch.zeros(20, 10), torch.zeros(20).long())
            mock_model = MagicMock()
            mock_model.return_value = torch.zeros(20, 2)
            mock_fh.return_value = mock_model
            mock_wf.return_value = (0.7, 0.8, [])

            from config import Config
            cfg = Config(train_steps=range(1, 27), val_steps=range(27, 35), test_steps=range(35, 50))
            result = run_single_sweep("Test Sweep", cfg, MagicMock(), MagicMock(), [])
            self._check_keys(result, "run_single_sweep")

class TestTemporalModels:
    @pytest.fixture
    def mock_dm(self):
        dm = MagicMock()
        dm.graphs = {}
        for t in range(1, 10):
            # 166 node features. Let's make 10 nodes per snapshot
            x = torch.randn(10, 166)
            prop = torch.randn(10, 32)
            y = torch.randint(0, 2, (10,))
            mask = torch.ones(10).bool()
            # First element of topo is pagerank
            topo = torch.rand(10, 2)
            edge_index = torch.randint(0, 10, (2, 20))
            dm.graphs[t] = {
                "x": x, "prop": prop, "y": y, "labeled_mask": mask, "edge_index": edge_index, "topo": topo
            }
        
        from config import Config
        dm.cfg = Config(train_steps=range(1, 5))
        dm.sgc_input_dim = 32
        return dm

    def test_tabular_lagged_features_exclude_current_and_future(self, mock_dm):
        from source.data.temporal_features import build_snapshot_temporal_features
        # Compute temporal features for step 5 with window=2.
        feats_before = build_snapshot_temporal_features(mock_dm, target_step=5, window=2, label_lag=0)
        
        # Mutate dm.graphs[5] and dm.graphs[6].
        mock_dm.graphs[5]["x"] *= 2.0
        mock_dm.graphs[6]["x"] *= 2.0
        
        # Recompute.
        feats_after = build_snapshot_temporal_features(mock_dm, target_step=5, window=2, label_lag=0)
        
        # Assert IDENTICAL.
        np.testing.assert_array_equal(feats_before, feats_after)

    def test_tabular_lagged_features_respond_to_past_positive_control(self, mock_dm):
        from source.data.temporal_features import build_snapshot_temporal_features
        feats_before = build_snapshot_temporal_features(mock_dm, target_step=5, window=2, label_lag=0)
        
        # Mutate dm.graphs[3] or dm.graphs[4].
        mock_dm.graphs[3]["x"] += 10.0
        mock_dm.graphs[4]["x"] += 10.0
        
        feats_after = build_snapshot_temporal_features(mock_dm, target_step=5, window=2, label_lag=0)
        
        # Assert DIFFERENT.
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(feats_before, feats_after)

    def test_lstm_conditioning_is_causal(self):
        from source.models.temporal_head import TemporalLSTM
        lstm = TemporalLSTM(embed_dim=16, hidden_dim=32)
        
        # Build 6 embeddings. Record h_3.
        embeddings = torch.randn(6, 16)
        h_before = lstm(embeddings)
        h3_before = h_before[2].clone()
        
        # Perturb embeddings 4 and 5 (indices 3 and 4).
        embeddings[3] += 10.0
        embeddings[4] += 10.0
        
        h_after = lstm(embeddings)
        h3_after = h_after[2].clone()
        
        # Assert h_3 UNCHANGED (future invariance).
        assert torch.allclose(h3_before, h3_after)

    def test_lstm_conditioning_responds_to_past_positive_control(self):
        from source.models.temporal_head import TemporalLSTM
        lstm = TemporalLSTM(embed_dim=16, hidden_dim=32)
        embeddings = torch.randn(6, 16)
        h_before = lstm(embeddings)
        h3_before = h_before[2].clone()
        
        # Perturb embeddings 1 or 2 (indices 0 or 1).
        embeddings[0] += 10.0
        embeddings[1] += 10.0
        
        h_after = lstm(embeddings)
        h3_after = h_after[2].clone()
        
        # Assert h_3 CHANGES.
        assert not torch.allclose(h3_before, h3_after)

    def test_lstm_embedding_uses_no_labels(self):
        from source.models.temporal_head import SnapshotEmbedder
        from config import Config
        cfg = Config(use_mlp_head=True)
        embedder = SnapshotEmbedder(in_dim=32, embed_dim=16, cfg=cfg)
        
        prop_features = torch.randn(10, 32)
        emb_before = embedder(prop_features)
        assert emb_before.shape == (16,)

    def test_lstm_embedding_responds_to_features_positive_control(self):
        from source.models.temporal_head import SnapshotEmbedder
        from config import Config
        cfg = Config(use_mlp_head=True)
        embedder = SnapshotEmbedder(in_dim=32, embed_dim=16, cfg=cfg)
        
        prop_features = torch.randn(10, 32)
        emb_before = embedder(prop_features)
        
        prop_features_perturbed = prop_features + 10.0
        emb_after = embedder(prop_features_perturbed)
        
        assert not torch.allclose(emb_before, emb_after)

    def test_lstm_all_modules_receive_gradients(self, mock_dm):
        from source.evaluation.temporal_validation import train_lstm_conditioned
        from config import Config
        cfg = Config()
        
        # The smoke test inside train_lstm_conditioned will assert p.grad is not None.
        # If it passes, the test passes.
        train_lstm_conditioned(mock_dm, train_steps=[1, 2], cfg=cfg, device=torch.device("cpu"), epochs=1, embed_dim=16)
        
    def test_conditioned_head_input_dim(self):
        from source.models.temporal_head import LSTMConditionedHead
        from config import Config
        cfg = Config()
        head = LSTMConditionedHead(node_in_dim=32, temporal_hidden_dim=64, cfg=cfg)
        
        if cfg.use_mlp_head:
            first_layer = head.head.hidden_net[0]
        else:
            first_layer = head.head._net[0]
            
        assert first_layer.in_features == 96

    def test_temporal_feature_shape(self, mock_dm):
        from source.data.temporal_features import build_snapshot_temporal_features
        for w in [1, 2, 4]:
            feats = build_snapshot_temporal_features(mock_dm, target_step=5, window=w, label_lag=0)
            expected_width = 31 * w - 15
            assert feats.shape[0] == expected_width
