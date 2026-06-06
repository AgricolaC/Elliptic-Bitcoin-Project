import numpy as np
import pandas as pd
from typing import Tuple, List


def _validate_temporal_edges(df: pd.DataFrame, df_edge: pd.DataFrame) -> None:
    """
    Validate that no edge crosses a timestep boundary and no orphan txIds exist.

    Defensive Notes (W3 fix):
        Old code used dict.map() which returns NaN for missing keys.
        NaN != NaN is False in NumPy, so orphan-edge cross-temporal violations
        were silently swallowed. This function now asserts NaN absence first,
        making orphan detection explicit and trustworthy.

    Args:
        df:      Node DataFrame with columns [txId, ts, ...].
        df_edge: Edge DataFrame with columns [txId1, txId2].

    Raises:
        AssertionError: If any edge endpoint is absent from df ("orphan" edge),
                        or if any edge spans two different timesteps.
    """
    ts_of = dict(zip(df.txId, df.ts))

    # LEAKAGE GUARD W3: map timestamps; preserve NaN for absent txIds.
    src_ts = df_edge.txId1.map(ts_of)
    dst_ts = df_edge.txId2.map(ts_of)

    # Step 1: detect orphan edges BEFORE the cross-temporal check.
    # Old code skipped this; NaN != NaN == False hid orphan violations.
    n_orphan_src = int(src_ts.isna().sum())
    n_orphan_dst = int(dst_ts.isna().sum())
    assert n_orphan_src == 0 and n_orphan_dst == 0, (
        f"orphan edge endpoints detected: "
        f"{n_orphan_src} txId1 and {n_orphan_dst} txId2 values not found in node DataFrame. "
        f"All edge endpoints must exist in df before the temporal guard runs."
    )

    # Step 2: now safe to compare — no NaN can hide a violation.
    cross_temporal = int(np.sum(src_ts.values != dst_ts.values))
    assert cross_temporal == 0, (
        f"temporal edge guard: {cross_temporal} cross-temporal edges detected. "
        f"All edges must connect nodes within the same timestep."
    )


import os, glob
import kagglehub

def download_and_load_data() -> Tuple[pd.DataFrame, pd.DataFrame, int, List[str]]:
    """Download and load Elliptic dataset from Kaggle."""
    DATA_ROOT = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    def _find(name):
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
    
    class_map = {"1": 1, "2": 0, "unknown": -1}
    df_class["label"] = df_class["class"].astype(str).map(class_map)
    df = df_feat.merge(df_class, on="txId", how="left")
    
    _validate_temporal_edges(df, df_edge)
    
    return df, df_edge, len(feature_cols), feature_cols
