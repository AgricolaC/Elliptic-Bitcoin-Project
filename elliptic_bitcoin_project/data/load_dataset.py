import os
import glob
import pandas as pd
import numpy as np
import kagglehub
from typing import Tuple, List

def download_and_load_data() -> Tuple[pd.DataFrame, pd.DataFrame, int, List[str]]:
    """
    Downloads dataset from Kaggle and performs schema validation.
    Returns:
        df: DataFrame with node features and labels.
        df_edge: DataFrame with edge list.
        NODE_FEATURE_DIM: Number of raw node features (166).
        feature_cols: List of column names corresponding to the features.
    """
    DATA_ROOT = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    
    def _find(name: str) -> str:
        hits = glob.glob(os.path.join(DATA_ROOT, "**", name), recursive=True)
        assert hits, f"Could not locate {name} under {DATA_ROOT}"
        return hits[0]

    feat_path  = _find("elliptic_txs_features.csv")
    class_path = _find("elliptic_txs_classes.csv")
    edge_path  = _find("elliptic_txs_edgelist.csv")

    df_feat  = pd.read_csv(feat_path, header=None)
    df_class = pd.read_csv(class_path)
    df_edge  = pd.read_csv(edge_path)

    df_feat = df_feat.rename(columns={0: "txId", 1: "ts"})
    feature_cols = list(df_feat.columns[1:])
    NODE_FEATURE_DIM = len(feature_cols)

    # Class mapping: illicit=1, licit=0, unknown=-1
    class_map = {"1": 1, "2": 0, "unknown": -1}
    df_class["label"] = df_class["class"].astype(str).map(class_map)
    df = df_feat.merge(df_class[["txId", "label"]], on="txId", how="left")
    df["label"] = df["label"].fillna(-1).astype(int)

    # SCHEMA GUARDS
    assert df.shape[0] == 203_769, f"Expected 203,769 nodes, got {df.shape[0]}"
    assert df_edge.shape[0] == 234_355, f"Expected 234,355 edges, got {df_edge.shape[0]}"
    assert NODE_FEATURE_DIM == 166, f"Expected 166 features, got {NODE_FEATURE_DIM}"
    assert df["ts"].min() == 1 and df["ts"].max() == 49, "Time steps must span 1..49"
    
    # TEMPORAL LEAKAGE GUARD
    ts_of = dict(zip(df.txId, df.ts))
    src_ts = df_edge.txId1.map(ts_of).values
    dst_ts = df_edge.txId2.map(ts_of).values
    cross_temporal = int(np.sum(src_ts != dst_ts))
    assert cross_temporal == 0, "Schema violation: temporal edges detected!"

    return df, df_edge, NODE_FEATURE_DIM, feature_cols
