import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import pytest

from config import Config
from data.build_graph import EllipticDataModule
from analysis.eda import plot_degree_distributions, plot_raw_feature_manifold, plot_feature_correlation

def _make_minimal_dm():
    """Build a tiny synthetic EllipticDataModule for EDA tests."""
    n_nodes, n_ts, n_feat = 60, 6, 10
    rng = np.random.default_rng(42)
    rows = []
    for t in range(1, n_ts + 1):
        for i in range(n_nodes):
            row = {"txId": t * 1000 + i, "ts": t, "label": rng.choice([0, 1, -1])}
            for f in range(n_feat):
                row[f"f{f}"] = rng.normal(0, 1.0)
            rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = [f"f{f}" for f in range(n_feat)]

    # Edge list: t=1 has empty edges. t>1 has sparse edges.
    edge_rows = []
    for t in range(2, n_ts + 1):
        ids = [t * 1000 + i for i in range(n_nodes)]
        for i in range(0, n_nodes - 1, 3):
            edge_rows.append({"txId1": ids[i], "txId2": ids[i + 1]})
    df_edge = pd.DataFrame(edge_rows)

    cfg = Config(use_topology=True, train_steps=range(1, 4), test_steps=range(4, n_ts + 1))
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    return dm

class TestEDAConstraints:
    
    def test_degree_distribution_handles_empty_edges(self):
        """A timestep with zero edges must not crash (log(0) / empty bincount guard)."""
        dm = _make_minimal_dm()
        # t=1 has NO edges in our synthetic dataset. This should not crash.
        try:
            plot_degree_distributions(dm, t=1)
        except Exception as e:
            pytest.fail(f"plot_degree_distributions crashed on empty edge index: {e}")

    def test_feature_eda_uses_raw_not_propagated(self, monkeypatch):
        """
        Feature-space functions must read the raw (166 or base+topo) matrix, 
        NEVER the SGC propagated tensor.
        """
        dm = _make_minimal_dm()
        
        # We patch PCA so it just records the shape of the data passed to it, then we assert
        recorded_shapes = []
        
        from sklearn.decomposition import PCA
        original_fit_transform = PCA.fit_transform
        
        def mock_fit_transform(self, X):
            recorded_shapes.append(X.shape)
            return original_fit_transform(self, X)
            
        monkeypatch.setattr(PCA, "fit_transform", mock_fit_transform)
        
        plot_raw_feature_manifold(dm, t=2)
        
        assert len(recorded_shapes) > 0
        passed_shape = recorded_shapes[0]
        
        # The passed data must be the raw feature dim (10 base + 2 topo = 12),
        # not the propagated dim which would be 12 * (K+1) = 36.
        assert passed_shape[1] == dm.feature_dim, f"Expected {dm.feature_dim} features, got {passed_shape[1]}. Circular EDA bug!"
        assert passed_shape[1] < dm.sgc_input_dim

    def test_correlation_fitted_on_train_only(self):
        """
        Correlation/effective-rank computation must never touch test-step rows.
        """
        dm = _make_minimal_dm()
        
        # Corrupt the test-set raw features to be all NaN
        for t in dm.cfg.test_steps:
            dm.graphs[t]["x"][:] = float('nan')
            
        try:
            plot_feature_correlation(dm)
        except ValueError:
            pytest.fail("plot_feature_correlation accessed test data, causing NaN crash! Leakage!")

if __name__ == "__main__":
    t = TestEDAConstraints()
    t.test_degree_distribution_handles_empty_edges()
    
    # Simple monkeypatch object for the second test
    class MockPatch:
        def setattr(self, obj, attr, val):
            setattr(obj, attr, val)
    t.test_feature_eda_uses_raw_not_propagated(MockPatch())
    t.test_correlation_fitted_on_train_only()
    print("ALL TESTS PASSED")
