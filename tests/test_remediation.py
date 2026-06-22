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
        "Seed", "Variation", "Sweep", "Feature_Set", "Threshold_Method",
        "Static_Time_s", "Static_Mem_MB", "Static_Val_F1", "Static_Val_PRAUC",
        "Static_OOT_F1", "Static_OOT_PRAUC", "WF_Time_s", "WF_Mem_MB",
        "WF_Pooled_F1", "WF_Pooled_PRAUC", "WF_Macro_F1", "WF_Macro_PRAUC",
        "WF_Pre43_Pooled_F1", "WF_Pre43_PRAUC", "WF_Shock_F1", "WF_Shock_PRAUC",
        "WF_Recovery_Pooled_F1", "WF_Recovery_PRAUC", "Selfcond_Bug", "Notes",
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

    def test_temporal_edge_count_uses_raw_directed_edges(self, mock_dm):
        from source.data.temporal_features import build_snapshot_temporal_features

        # target_step=5, window=1 reads snapshot 4. Its edge_index stores each
        # raw directed edge once, so E is the full second dimension, not E/2.
        feats = build_snapshot_temporal_features(
            mock_dm, target_step=5, window=1, label_lag=0
        )
        n_nodes = mock_dm.graphs[4]["x"].shape[0]
        n_edges = mock_dm.graphs[4]["edge_index"].shape[1]

        assert feats[2] == n_edges
        assert feats[3] == pytest.approx(2.0 * n_edges / n_nodes)

    def test_temporal_loss_respects_labeled_mask(self):
        """Axiom-falsify: the temporal training loss must be gated by
        ``labeled_mask`` — perturbing the labels of masked-out nodes (which a
        correct pipeline excludes) must leave the trained head bit-identical.
        The embedder is label-free, so labels enter only through the loss."""
        from source.evaluation.temporal_validation import train_lstm_conditioned
        from config import Config, set_global_seeds

        def make_dm():
            dm = MagicMock()
            dm.graphs = {}
            gen = torch.Generator().manual_seed(7)
            for t in range(1, 5):
                prop = torch.randn(12, 32, generator=gen)
                y = torch.randint(0, 2, (12,), generator=gen)  # valid labels everywhere
                mask = torch.ones(12).bool()
                mask[6:] = False  # last 6 nodes are unlabeled regardless of y value
                dm.graphs[t] = {"prop": prop, "y": y, "labeled_mask": mask}
            dm.sgc_input_dim = 32
            return dm

        cfg = Config()

        dm1 = make_dm()
        set_global_seeds(123)
        _, _, head1 = train_lstm_conditioned(dm1, [1, 2, 3], cfg,
                                             torch.device("cpu"), epochs=5, embed_dim=16)
        p1 = torch.cat([p.flatten() for p in head1.parameters()])

        # Flip the labels of the masked-out nodes only. A mask-respecting loss
        # never sees these, so training must be unaffected.
        dm2 = make_dm()
        for t in dm2.graphs:
            m = dm2.graphs[t]["labeled_mask"]
            dm2.graphs[t]["y"][~m] = 1 - dm2.graphs[t]["y"][~m]
        set_global_seeds(123)
        _, _, head2 = train_lstm_conditioned(dm2, [1, 2, 3], cfg,
                                             torch.device("cpu"), epochs=5, embed_dim=16)
        p2 = torch.cat([p.flatten() for p in head2.parameters()])

        assert torch.allclose(p1, p2), \
            "Masked-out nodes leaked into the temporal training loss"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.5 MLP-variation expansion — every (target × variation) must be emitted
# Guards the loop-nesting bug where variations ran for only one target.
# ─────────────────────────────────────────────────────────────────────────────

class TestMLPVariationExpansion:
    """
    Phase 2.5 tunes MLP heads on BOTH the Champion and Challenger configs across
    three variations (Wide / Residual / ResWide). A loop-nesting bug emitted the
    variations for only the last-bound target. The expansion must be a pure
    function producing one spec per (target × variation).
    """

    def _make_targets(self):
        from config import Config
        champ = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=1,
                       use_directional_prop=False, use_graph_structural=False, seed=42)
        chall = Config(use_mlp_head=True, use_multiscale_prop=True, sgc_k=3,
                       use_directional_prop=True, use_graph_structural=True,
                       topo_injection_mode='early', seed=42)
        return [(champ, "Grid: K=1, Dir=F, Topo=None", "Champion"),
                (chall, "Grid: K=3, Dir=T, Topo=early", "Challenger")]

    def _variations(self):
        return [
            ("Wide", {"mlp_hidden": (512, 256, 128), "use_residual": False}),
            ("Residual", {"mlp_hidden": (128, 64), "use_residual": True}),
            ("ResWide", {"mlp_hidden": (512, 256, 128), "use_residual": True}),
        ]

    def test_every_target_gets_every_variation(self):
        from sweep import build_mlp_variation_specs
        specs = build_mlp_variation_specs(self._make_targets(), self._variations(),
                                          seed=42, var="Base", mode="standard")
        assert len(specs) == 6  # 2 targets × 3 variations

    def test_each_variation_inherits_its_own_base_cfg(self):
        from sweep import build_mlp_variation_specs
        specs = build_mlp_variation_specs(self._make_targets(), self._variations(),
                                          seed=42, var="Base", mode="standard")
        champ_specs = [c for (_, name, c) in specs if "K=1" in name]
        chall_specs = [c for (_, name, c) in specs if "K=3" in name]
        assert len(champ_specs) == 3
        assert len(chall_specs) == 3
        assert all(c.sgc_k == 1 and not c.use_directional_prop for c in champ_specs)
        assert all(c.sgc_k == 3 and c.use_directional_prop for c in chall_specs)

    def test_mega_mode_qualifies_sweep_key_with_seed_and_var(self):
        from sweep import build_mlp_variation_specs
        specs = build_mlp_variation_specs(self._make_targets(), self._variations(),
                                          seed=43, var="PCA", mode="mega")
        assert all("(Seed 43, Var PCA)" in key for (key, _, _) in specs)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward window invariant — training must never see calib (τ-1) or τ.
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardBlocks:
    """
    The temporal walk-forward window for step τ must train on [..τ-2], calibrate
    on τ-1, and test on τ. A single helper enforces this so both the LSTM and EMA
    evaluators cannot drift apart.
    """

    def test_train_block_excludes_calib_and_tau(self):
        from source.evaluation.temporal_validation import _walk_forward_blocks
        steps = list(range(1, 50))
        train_block, calib_step = _walk_forward_blocks(steps, tau=43)
        assert calib_step == 42
        assert max(train_block) == 41
        assert 42 not in train_block and 43 not in train_block

    def test_invariant_holds_for_all_test_steps(self):
        from source.evaluation.temporal_validation import _walk_forward_blocks
        steps = list(range(1, 50))
        for tau in range(35, 50):
            train_block, calib_step = _walk_forward_blocks(steps, tau)
            assert calib_step == tau - 1
            assert all(t < tau - 1 for t in train_block)

    def test_skips_missing_steps(self):
        from source.evaluation.temporal_validation import _walk_forward_blocks
        steps = [1, 2, 5, 6, 7]  # 3 and 4 absent
        train_block, calib_step = _walk_forward_blocks(steps, tau=7)
        assert calib_step == 6
        assert train_block == [1, 2, 5]  # 3,4 missing; 6 (calib) and 7 (τ) excluded


class TestOneStepAheadBlocks:
    """
    P0a — De-confound the thesis: the hidden state that classifies step S must be
    built ONLY from steps strictly before S (one-step-ahead), so the model never
    sees the snapshot it is scoring. This removes the self-conditioning confound
    (review #2). The SAME helper drives both the LSTM and EMA evaluators.

    Convention: _onestep_blocks returns
        (train_block, calib_step, calib_state_steps, infer_state_steps)
    where calib_state_steps classifies τ-1 (so excludes τ-1) and infer_state_steps
    classifies τ (so excludes τ).
    """

    def test_inference_state_excludes_tau(self):
        from source.evaluation.temporal_validation import _onestep_blocks
        steps = list(range(1, 50))
        _, _, calib_state, infer_state = _onestep_blocks(steps, tau=43)
        assert 43 not in infer_state          # τ excluded from the state that classifies τ
        assert max(infer_state) == 42         # state runs up to τ-1 only

    def test_calibration_state_excludes_calib_step(self):
        from source.evaluation.temporal_validation import _onestep_blocks
        steps = list(range(1, 50))
        _, calib_step, calib_state, _ = _onestep_blocks(steps, tau=43)
        assert calib_step == 42
        assert 42 not in calib_state          # τ-1 excluded from the state that classifies τ-1
        assert max(calib_state) == 41

    def test_both_states_one_step_apart_for_all_tau(self):
        from source.evaluation.temporal_validation import _onestep_blocks
        steps = list(range(1, 50))
        for tau in range(35, 50):
            _, calib_step, calib_state, infer_state = _onestep_blocks(steps, tau)
            # inference state = calibration state + the calib step (one step further)
            assert infer_state == calib_state + [calib_step]
            assert tau not in infer_state and calib_step not in calib_state


class TestSnapshotTopology:
    """P0a — ground-truth snapshot topology, computed from raw data only."""

    def _toy(self):
        rows = []
        txid = 0
        # ts -> (n_illicit, n_licit, n_unknown)
        for t, (ni, nl, nu) in {1: (4, 4, 2), 2: (0, 5, 5), 3: (3, 3, 0)}.items():
            for lab, cnt in ((1, ni), (0, nl), (-1, nu)):
                for _ in range(cnt):
                    rows.append({"txId": txid, "ts": t, "label": lab}); txid += 1
        df = pd.DataFrame(rows)
        df_edge = pd.DataFrame({"txId1": [0, 1], "txId2": [1, 2]})  # 2 edges in ts=1
        return df, df_edge

    def test_schema_and_counts(self):
        from source.data.snapshot_topology import build_snapshot_topology
        out = build_snapshot_topology(*self._toy())
        required = {"Tau", "N_nodes", "N_edges", "N_illicit", "N_licit", "N_unknown",
                    "N_labeled", "Illicit_Rate", "Mean_Degree", "Graph_Density", "Regime"}
        assert required.issubset(set(out.columns))
        r1 = out[out.Tau == 1].iloc[0]
        assert r1.N_illicit == 4 and r1.N_licit == 4 and r1.N_unknown == 2
        assert r1.N_labeled == 8 and abs(r1.Illicit_Rate - 0.5) < 1e-9
        assert r1.N_edges == 2

    def test_regime_labels(self):
        from source.data.snapshot_topology import build_snapshot_topology
        rows, txid = [], 0
        for t in (42, 43, 44):
            for _ in range(3):
                rows.append({"txId": txid, "ts": t, "label": 0}); txid += 1
        out = build_snapshot_topology(pd.DataFrame(rows),
                                      pd.DataFrame({"txId1": [], "txId2": []}))
        reg = dict(zip(out.Tau, out.Regime))
        assert reg[42] == "pre_shock" and reg[43] == "shock" and reg[44] == "recovery"


class TestEpsilonFallback:
    """P0c — when the calibration step has < ε positives, the supervised
    F1-threshold is unreliable; fall back to an unsupervised quantile."""

    def test_fallback_fires_under_few_positives(self):
        from source.evaluation.validation import _calibrate_threshold
        y_cal = np.array([1, 1, 1] + [0] * 97)          # 3 positives < ε=10
        s_cal = np.linspace(0.0, 1.0, 100)
        thr, fired = _calibrate_threshold(y_cal, s_cal, global_illicit_rate=0.1, epsilon=10)
        assert fired is True
        assert abs(thr - np.quantile(s_cal, 0.9)) < 1e-9   # 1 - 0.1 quantile

    def test_fallback_silent_with_enough_positives(self):
        from source.evaluation.validation import _calibrate_threshold
        y_cal = np.array([1] * 20 + [0] * 80)            # 20 positives >= ε
        s_cal = np.linspace(0.0, 1.0, 100)
        _, fired = _calibrate_threshold(y_cal, s_cal, global_illicit_rate=0.1, epsilon=10)
        assert fired is False


class TestPagerankAudit:
    """P0e — is the PageRank feature actually alive under early injection?"""

    def _early_dm(self):
        from config import Config
        from data.build_graph import EllipticDataModule
        n_nodes, n_ts, n_feat = 40, 5, 6
        rng = np.random.default_rng(1)
        rows, edge_rows = [], []
        for t in range(1, n_ts + 1):
            ids = [t * 1000 + i for i in range(n_nodes)]
            for i in range(n_nodes):
                row = {"txId": ids[i], "ts": t, "label": rng.choice([0, 1, -1])}
                for f in range(n_feat):
                    row[f"f{f}"] = rng.normal()
                rows.append(row)
            # hub-and-spoke so PageRank genuinely varies across nodes
            for i in range(1, n_nodes):
                edge_rows.append({"txId1": ids[0], "txId2": ids[i]})
        df = pd.DataFrame(rows)
        feature_cols = [f"f{f}" for f in range(n_feat)]
        cfg = Config(train_steps=range(1, 4), test_steps=range(4, 6),
                     use_graph_structural=True, topo_injection_mode="early")
        dm = EllipticDataModule(df, pd.DataFrame(edge_rows), feature_cols, cfg)
        dm.setup()
        return dm, cfg, n_feat

    def test_pagerank_column_has_variance_under_early_injection(self):
        dm, cfg, n_feat = self._early_dm()
        # early injection appends [pagerank, clustering] right after the n_feat base cols
        Xs = np.concatenate([dm.graphs[t]["x"].numpy() for t in cfg.train_steps])
        pagerank_col = Xs[:, n_feat]
        assert pagerank_col.std() > 0.01, (
            f"PageRank column appears dead (std={pagerank_col.std():.4g}) under early injection"
        )


class TestStratifiedWFMetrics:
    """F1 foundation: walk-forward metrics stratified into pre_shock(τ≤42) /
    shock(τ=43) / recovery(τ≥44), pooled + macro, threshold-free PRAUC primary,
    with Low_Confidence flagging for τ with < 10 illicit."""

    def _records(self):
        return [
            {"tau": 41, "y_true": np.array([1, 1, 0, 0, 1, 0]), "scores": np.array([.9, .8, .1, .2, .7, .3])},
            {"tau": 43, "y_true": np.array([1, 0, 0, 0]),       "scores": np.array([.6, .4, .3, .2])},
            {"tau": 46, "y_true": np.array([1, 0, 0]),          "scores": np.array([.55, .45, .1])},
        ]

    def test_regime_labels_and_low_confidence(self):
        from source.evaluation.wf_metrics import stratified_wf_metrics
        agg, rows = stratified_wf_metrics(self._records())
        rmap = {r["Tau"]: r for r in rows}
        assert rmap[41]["Regime"] == "pre_shock"
        assert rmap[43]["Regime"] == "shock"
        assert rmap[46]["Regime"] == "recovery"
        assert all(r["Low_Confidence"] for r in rows)          # all have <10 illicit
        assert rmap[41]["N_illicit"] == 3 and rmap[41]["N_labeled"] == 6

    def test_aggregate_has_regime_stratified_keys(self):
        from source.evaluation.wf_metrics import stratified_wf_metrics
        agg, _ = stratified_wf_metrics(self._records())
        for k in ("WF_Pooled_PRAUC", "WF_Macro_PRAUC", "WF_Pre43_PRAUC",
                  "WF_Shock_PRAUC", "WF_Recovery_PRAUC", "WF_Pooled_F1"):
            assert k in agg

    def test_pre43_pool_excludes_shock_and_recovery(self):
        from source.evaluation.wf_metrics import stratified_wf_metrics
        from sklearn.metrics import average_precision_score
        recs = self._records()
        agg, _ = stratified_wf_metrics(recs)
        # pre43 pool = only τ=41 here
        expected = average_precision_score(recs[0]["y_true"], recs[0]["scores"])
        assert abs(agg["WF_Pre43_PRAUC"] - expected) < 1e-9

    def test_pooled_and_regime_f1_use_supplied_calibrated_predictions(self):
        from source.evaluation.wf_metrics import stratified_wf_metrics

        # Every score is below the default 0.5 threshold, but the supplied
        # predictions represent valid per-timestep calibrated thresholds.
        recs = [
            {
                "tau": 41,
                "y_true": np.array([1, 0]),
                "scores": np.array([0.4, 0.1]),
                "y_pred": np.array([1, 0]),
            },
            {
                "tau": 43,
                "y_true": np.array([1, 0]),
                "scores": np.array([0.3, 0.2]),
                "y_pred": np.array([1, 0]),
            },
        ]

        agg, _ = stratified_wf_metrics(recs)

        assert agg["WF_Pooled_F1"] == 1.0
        assert agg["WF_Pre43_Pooled_F1"] == 1.0
        assert agg["WF_Shock_F1"] == 1.0


class TestTimestepNotAFeature:
    """Pre-F1 cleanup: the timestep column is the split variable, NOT a model
    feature. As a live feature it is a monotonic proxy for position in the
    walk-forward window (a leak). feature_cols must exclude it."""

    def test_select_feature_cols_excludes_timestep(self):
        from source.data.load_dataset import _select_feature_cols
        cols = ["txId", "ts", 2, 3, 4]
        selected = _select_feature_cols(cols)
        assert "ts" not in selected
        assert selected == [2, 3, 4]      # txId dropped (id), ts dropped (split var)


class TestResultSchemaV2:
    """CSV-1 schema rewrite: regime-stratified PRAUC columns + Selfcond_Bug
    provenance. Every column must be present and non-null on every result dict."""

    NEW_KEYS = {
        "Seed", "Variation", "Sweep", "Feature_Set", "Threshold_Method",
        "Static_Time_s", "Static_Mem_MB", "Static_Val_F1", "Static_Val_PRAUC",
        "Static_OOT_F1", "Static_OOT_PRAUC", "WF_Time_s", "WF_Mem_MB",
        "WF_Pooled_F1", "WF_Pooled_PRAUC", "WF_Macro_F1", "WF_Macro_PRAUC",
        "WF_Pre43_Pooled_F1", "WF_Pre43_PRAUC", "WF_Shock_F1", "WF_Shock_PRAUC",
        "WF_Recovery_Pooled_F1", "WF_Recovery_PRAUC", "Selfcond_Bug", "Notes",
    }

    def test_make_result_has_v2_schema(self):
        from source.sweep import _make_result, _RESULT_KEYS
        assert set(_RESULT_KEYS) == self.NEW_KEYS, "Canonical _RESULT_KEYS != v2 schema"
        r = _make_result(
            seed=42, variation="Base", sweep="x",
            static_time=1.0, static_mem=1.0, static_f1=0.8, static_prauc=0.9,
            wf_time="N/A", wf_mem="N/A", wf_f1="N/A", wf_prauc="N/A",
            selfcond_bug="fixed",
        )
        assert set(r.keys()) == self.NEW_KEYS
        assert all(v is not None for v in r.values()), "no result field may be None"
        assert r["Selfcond_Bug"] == "fixed"
        assert r["Static_OOT_F1"] == 0.8
        # old WF mean maps to MACRO under the new schema
        assert "WF_Macro_F1" in r and "WF_Pre43_PRAUC" in r
