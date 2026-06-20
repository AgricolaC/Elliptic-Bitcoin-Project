import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict, Any
from config import Config
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import networkx as nx

def topological_features(edge_index: torch.Tensor, n: int) -> np.ndarray:
    """PageRank + clustering coefficient per node."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_index.t().tolist())
    pr = nx.pagerank(G, alpha=0.85, max_iter=200) if G.number_of_edges() else {i: 1.0/n for i in range(n)}
    cl = nx.clustering(G.to_undirected())
    feats = np.array([[pr.get(i, 0.0), cl.get(i, 0.0)] for i in range(n)], dtype=np.float32)
    assert feats.shape == (n, 2), f"Topology feat shape {feats.shape} != ({n}, 2)"
    return feats

try:
    from models.layers import sgc_propagate
except ImportError:
    sgc_propagate = None


def reindex_timestep(
    sub_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map non-contiguous global txIds -> contiguous local ids [0..n-1].

    Defensive Notes:
        - Asserts edge_index shape [2, E].
        - Asserts all local indices are in [0, n-1].
        - Asserts node/feature/label count consistency.
    """
    tx_ids = sub_df.txId.values
    id_to_local = {int(g): i for i, g in enumerate(tx_ids)}
    n = len(tx_ids)

    in_slice = edges_df.txId1.isin(id_to_local) & edges_df.txId2.isin(id_to_local)
    e = edges_df[in_slice]
    if len(e):
        src = e.txId1.map(id_to_local).values.astype(np.int64)
        dst = e.txId2.map(id_to_local).values.astype(np.int64)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # SHAPE GUARD
    assert edge_index.shape[0] == 2, \
        f"edge_index must be [2,E], got {tuple(edge_index.shape)}"
    if edge_index.numel():
        assert int(edge_index.min()) >= 0 and int(edge_index.max()) < n, \
            "Local edge index out of bounds — re-indexing failed."

    X = sub_df[feature_cols].values.astype(np.float32)
    y = sub_df.label.values.astype(np.int64)
    assert X.shape[0] == n == len(y), "Node/feature/label count mismatch."
    return edge_index, X, y, tx_ids


class EllipticDataModule:
    """
    Builds per-timestep graph tensors with feature injections.

    Fix W1: A second StandardScaler pass is fitted on train data AFTER topology
            and SVD-reconstruction columns are appended, so every column in x
            operates on the same variance scale before entering SGC propagation.

    Fix W6: setup() now internally runs SGC propagation and sets sgc_input_dim.
            Callers never need to assign dm.sgc_input_dim or call sgc_propagate
            externally.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        edges_df: pd.DataFrame,
        feature_cols: List[str],
        cfg: Config,
    ):
        self.df = df
        self.edges_df = edges_df
        self.feature_cols = feature_cols
        self.cfg = cfg
        self.graphs: Dict[int, Dict[str, Any]] = {}

        # LEAKAGE GUARD: two independent scalers — each fitted on train only.
        # scaler_base: fitted on raw 166 features.
        # scaler_aug:  fitted on [base | topology] (W1 fix).
        # scaler_topo: fitted on topology only.
        self.scaler_base = StandardScaler()
        self.scaler_aug  = StandardScaler()
        self.scaler_topo = StandardScaler()
        self.scaler_prop = StandardScaler()   # P2-B: post-propagation rescaling
        self.feature_dim = len(feature_cols)
        self.sgc_input_dim: int = -1   # W6: will be set inside setup()

    def setup(self) -> "EllipticDataModule":
        """
        Build graphs, scale features, optionally inject topology,
        apply a second scaler pass (W1 fix), run SGC propagation (W6 fix).

        Defensive Notes:
            - Both scalers fitted exclusively on train_steps (no leakage).
            - sgc_input_dim set from actual propagated tensor shape.
            - Shape asserts at every injection boundary.
        """
        c = self.cfg
        ts_min = min(min(c.train_steps), min(c.test_steps))
        ts_max = max(max(c.train_steps), max(c.test_steps))

        # ── Step 1: Build per-timestep raw graphs ──────────────────────────────
        for t in range(ts_min, ts_max + 1):
            sub = self.df[self.df.ts == t]
            if len(sub) == 0:
                continue
            ei, X, y, txids = reindex_timestep(sub, self.edges_df, self.feature_cols)
            self.graphs[t] = dict(
                edge_index=ei,
                x_np=X,
                y=torch.tensor(y),
                labeled_mask=torch.tensor(y != -1),
                n=len(y),
                txids=txids,
            )

        # ── Step 2: Base scaler — fitted on train 166-dim features only ────────
        # LEAKAGE GUARD: scaler_base never sees test-step data.
        train_X_raw = np.concatenate(
            [self.graphs[t]["x_np"] for t in c.train_steps if t in self.graphs], axis=0
        )
        self.scaler_base.fit(train_X_raw)
        for t in self.graphs:
            self.graphs[t]["x_np"] = self.scaler_base.transform(self.graphs[t]["x_np"])

        # ── Step 3: Optional topology injection ────────────────────────────────
        if c.use_graph_structural:
            for t in self.graphs:
                ei = self.graphs[t]["edge_index"]
                n  = self.graphs[t]["n"]
                topo_feats = topological_features(ei, n)
                # SHAPE GUARD
                assert topo_feats.shape == (n, 2), \
                    f"t={t}: topology shape {topo_feats.shape} != ({n}, 2)"
                if c.topo_injection_mode == 'early':
                    self.graphs[t]["x_np"] = np.concatenate(
                        [self.graphs[t]["x_np"], topo_feats], axis=1
                    )
                else:
                    self.graphs[t]["topo_np"] = topo_feats

        # ── Step 5: Second scaler pass (W1 FIX) ───────────────────────────────
        # Rescales the full augmented feature matrix [base | topo] so
        # that topology (PageRank ~1e-4, clustering ~0..1) operate on the 
        # same variance scale as the base features.
        #
        # LEAKAGE GUARD: scaler_aug fitted on train_steps augmented data only.
        if c.use_graph_structural:
            if getattr(c, 'topo_injection_mode', 'late') == 'early':
                train_X_full = np.concatenate(
                    [self.graphs[t]["x_np"] for t in c.train_steps if t in self.graphs], axis=0
                )
                self.scaler_aug.fit(train_X_full)
                for t in self.graphs:
                    self.graphs[t]["x_np"] = self.scaler_aug.transform(self.graphs[t]["x_np"])
            else:
                train_topo_full = np.concatenate(
                    [self.graphs[t]["topo_np"] for t in c.train_steps if t in self.graphs], axis=0
                )
                self.scaler_topo.fit(train_topo_full)
                for t in self.graphs:
                    self.graphs[t]["topo_np"] = self.scaler_topo.transform(self.graphs[t]["topo_np"])

        # ── Step 6: Finalize tensors ───────────────────────────────────────────
        for t in self.graphs:
            x = torch.tensor(self.graphs[t]["x_np"], dtype=torch.float32)
            if c.use_graph_structural and getattr(c, 'topo_injection_mode', 'late') == 'late':
                topo = torch.tensor(self.graphs[t]["topo_np"], dtype=torch.float32)
                self.graphs[t]["topo"] = torch.nan_to_num(topo, nan=0.0, posinf=0.0, neginf=0.0)
            self.graphs[t]["x"] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_dim = self.graphs[ts_min]["x"].shape[1]

        # ── Step 7: SGC Propagation (W6 FIX) ──────────────────────────────────
        # Encapsulated here so callers never need to run the propagation loop
        # or assign dm.sgc_input_dim manually.
        if sgc_propagate is not None:
            for t in self.graphs:
                g = self.graphs[t]
                prop = sgc_propagate(
                    g["x"], g["edge_index"], c.sgc_k, c.use_multiscale_prop, c.use_directional_prop
                )
                if c.use_graph_structural and getattr(c, 'topo_injection_mode', 'late') == 'late':
                    prop = torch.cat([prop, g["topo"]], dim=1)
                g["prop"] = prop
            # SHAPE GUARD: verify dim from actual tensor
            sample_prop = self.graphs[ts_min]["prop"]
            
            if c.use_directional_prop:
                expected_dim = self.feature_dim * (1 + 3 * c.sgc_k)
            elif c.use_multiscale_prop:
                expected_dim = self.feature_dim * (c.sgc_k + 1)
            else:
                expected_dim = self.feature_dim
                
            if c.use_graph_structural and getattr(c, 'topo_injection_mode', 'late') == 'late':
                expected_dim += 2
                
            assert sample_prop.shape[1] == expected_dim, (
                f"sgc_input_dim mismatch: got {sample_prop.shape[1]}, "
                f"expected {expected_dim}"
            )
            self.sgc_input_dim = sample_prop.shape[1]
        else:
            # Fallback for test environments where layers is unavailable
            self.sgc_input_dim = self.feature_dim

        # ── Step 7.5: Post-propagation StandardScaler (P2-B hop rescaling) ─────
        # Each S^k X hop shrinks variance systematically. Without rescaling,
        # later hops enter the head at smaller scale and L1/weight-decay
        # penalties are effectively unequal across hops.
        # LEAKAGE GUARD: scaler_prop fitted on train_steps only.
        train_props = np.concatenate(
            [self.graphs[t]["prop"].numpy() for t in c.train_steps if t in self.graphs],
            axis=0,
        )
        self.scaler_prop.fit(train_props)
        for t in self.graphs:
            prop_np = self.scaler_prop.transform(self.graphs[t]["prop"].numpy())
            self.graphs[t]["prop"] = torch.tensor(prop_np, dtype=torch.float32)

        # ── Step 8: PCA Dimensionality Reduction ──────────────────────────────
        if getattr(c, 'use_pca', False) or getattr(c, 'use_ipca', False):
            print(f"[DataModule] Applying PCA (variance retained: {c.pca_variance}) to propagated features...")
            train_props = []
            for t in c.train_steps:
                if t in self.graphs:
                    train_props.append(self.graphs[t]["prop"].numpy())
            train_props_full = np.concatenate(train_props, axis=0)
            
            # Scikit-learn requires svd_solver='full' when n_components is a float between 0 and 1
            solver = 'full' if isinstance(c.pca_variance, float) else 'auto'
            pca_base = PCA(n_components=c.pca_variance, svd_solver=solver)
            pca_base.fit(train_props_full)
            n_components = pca_base.n_components_
            print(f"[DataModule] PCA selected {n_components} components to retain {c.pca_variance} variance.")
            
            if getattr(c, 'use_ipca', False):
                from sklearn.decomposition import IncrementalPCA
                b_size = max(n_components, 512)
                pca = IncrementalPCA(n_components=n_components, batch_size=b_size)
                pca.fit(train_props_full)
                self.ipca = pca
            else:
                pca = pca_base
            
            for t in self.graphs:
                prop_np = self.graphs[t]["prop"].numpy()
                self.graphs[t]["prop_raw"] = prop_np
                prop_pca = pca.transform(prop_np)
                self.graphs[t]["prop"] = torch.tensor(prop_pca, dtype=torch.float32)
                
            self.sgc_input_dim = n_components

        n_base = len(self.feature_cols)
        n_topo = 2 if c.use_graph_structural else 0
        print(
            f"[DataModule] feature_dim={self.feature_dim} "
            f"({n_base} base + {n_topo} graph_struct) | "
            f"sgc_input_dim={self.sgc_input_dim}"
        )
        return self
