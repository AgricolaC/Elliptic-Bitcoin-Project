import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict, Any
from config import Config
from features.network_metrics import topological_features
from features.reconstruction import SVDReconstructor
from sklearn.preprocessing import StandardScaler

def reindex_timestep(sub_df: pd.DataFrame, edges_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Map non-contiguous global txIds -> contiguous local ids [0..n-1]."""
    tx_ids = sub_df.txId.values
    id_to_local = {int(g): i for i, g in enumerate(tx_ids)}
    n = len(tx_ids)
    in_slice = edges_df.txId1.isin(id_to_local) & edges_df.txId2.isin(id_to_local)
    e = edges_df[in_slice]
    src = e.txId1.map(id_to_local).values.astype(np.int64)
    dst = e.txId2.map(id_to_local).values.astype(np.int64)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    # SHAPE/RANGE GUARDS
    assert edge_index.shape[0] == 2, f"edge_index must be [2,E], got {tuple(edge_index.shape)}"
    if edge_index.numel():
        assert int(edge_index.min()) >= 0 and int(edge_index.max()) < n, \
            "Local edge index out of bounds — re-indexing failed."
            
    X = sub_df[feature_cols].values.astype(np.float32)
    y = sub_df.label.values.astype(np.int64)
    assert X.shape[0] == n == len(y), "Node/feature/label count mismatch."
    return edge_index, X, y, tx_ids

class EllipticDataModule:
    """Builds per-timestep graph tensors with feature injections."""
    def __init__(self, df: pd.DataFrame, edges_df: pd.DataFrame, feature_cols: List[str], cfg: Config):
        self.df = df
        self.edges_df = edges_df
        self.feature_cols = feature_cols
        self.cfg = cfg
        self.graphs: Dict[int, Dict[str, Any]] = {}
        self.scaler = StandardScaler()
        self.svd = SVDReconstructor(cfg)
        self.feature_dim = len(feature_cols)

    def setup(self) -> "EllipticDataModule":
        c = self.cfg
        
        # 1) Build per-timestep raw graphs
        for t in range(1, 50):
            sub = self.df[self.df.ts == t]
            ei, X, y, txids = reindex_timestep(sub, self.edges_df, self.feature_cols)
            self.graphs[t] = dict(edge_index=ei, x_np=X, y=torch.tensor(y),
                                  labeled_mask=torch.tensor(y != -1),
                                  n=len(y), txids=txids)
                                  
        # 2) Standard Scaling (prevent temporal leakage)
        train_X_raw = np.concatenate([self.graphs[t]["x_np"] for t in c.train_steps], axis=0)
        self.scaler.fit(train_X_raw)
        
        for t in range(1, 50):
            self.graphs[t]["x_np"] = self.scaler.transform(self.graphs[t]["x_np"])

        # 3) Optional topology addition
        if c.use_topology:
            for t in range(1, 50):
                ei = self.graphs[t]["edge_index"]
                n = self.graphs[t]["n"]
                topo_feats = topological_features(ei, n)
                self.graphs[t]["x_np"] = np.concatenate([self.graphs[t]["x_np"], topo_feats], axis=1)

        # 4) SVD Reconstruction error
        if c.use_recon_error:
            # Fit on scaled train data
            train_X = np.concatenate([self.graphs[t]["x_np"] for t in c.train_steps], axis=0)
            self.svd.fit(train_X)
            for t in range(1, 50):
                err = self.svd.get_reconstruction_error(self.graphs[t]["x_np"])
                self.graphs[t]["x_np"] = np.concatenate([self.graphs[t]["x_np"], err], axis=1)

        # 5) Finalize tensors
        for t in range(1, 50):
            x = torch.tensor(self.graphs[t]["x_np"], dtype=torch.float32)
            self.graphs[t]["x"] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        self.feature_dim = self.graphs[1]["x"].shape[1]
        print(f"[DataModule] feature_dim={self.feature_dim} "
              f"({len(self.feature_cols)} base + {'2 topo ' if c.use_topology else ''}"
              f"{'+1 recon' if c.use_recon_error else ''})")
        return self
