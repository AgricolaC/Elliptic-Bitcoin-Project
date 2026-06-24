"""Ground-truth per-snapshot topology, computed from the raw node/edge tables
only — no model, no propagation. Feeds World-C analysis and the presentation's
dataset section. Label encoding: 1=illicit, 0=licit, -1=unknown.
"""
import numpy as np
import pandas as pd


def _regime(tau: int) -> str:
    if tau <= 42:
        return "pre_shock"
    if tau == 43:
        return "shock"
    return "recovery"


def build_snapshot_topology(df: pd.DataFrame, df_edge: pd.DataFrame) -> pd.DataFrame:
    """Return one row per timestep τ with node/edge counts, label counts,
    illicit rate, mean degree, density, and regime label.

    Edges are within-timestep (validated upstream), so each edge is attributed
    to the timestep of its endpoints.
    """
    ts_of = dict(zip(df.txId, df.ts))
    if len(df_edge) > 0:
        edge_ts = df_edge.txId1.map(ts_of)
        edges_per_ts = edge_ts.value_counts().to_dict()
    else:
        edges_per_ts = {}

    rows = []
    for tau in sorted(df.ts.unique()):
        sub = df[df.ts == tau]
        n_nodes = int(len(sub))
        n_illicit = int((sub.label == 1).sum())
        n_licit = int((sub.label == 0).sum())
        n_unknown = int((sub.label == -1).sum())
        n_labeled = n_illicit + n_licit
        n_edges = int(edges_per_ts.get(tau, 0))

        illicit_rate = (n_illicit / n_labeled) if n_labeled > 0 else np.nan
        mean_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
        density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0

        rows.append({
            "Tau": int(tau),
            "N_nodes": n_nodes,
            "N_edges": n_edges,
            "N_illicit": n_illicit,
            "N_licit": n_licit,
            "N_unknown": n_unknown,
            "N_labeled": n_labeled,
            "Illicit_Rate": illicit_rate,
            "Mean_Degree": mean_degree,
            "Graph_Density": density,
            "Regime": _regime(int(tau)),
        })
    return pd.DataFrame(rows).sort_values("Tau").reset_index(drop=True)


if __name__ == "__main__":
    import os
    from source.config import OUTPUT_DIR
    from source.data.load_dataset import download_and_load_data

    print("Loading data...")
    df, df_edge, _, _ = download_and_load_data()
    print("Building snapshot topology...")
    topo = build_snapshot_topology(df, df_edge)
    out_path = os.path.join(OUTPUT_DIR, "snapshot_topology.csv")
    topo.to_csv(out_path, index=False)
    print(f"Saved {len(topo)} snapshots to {out_path}")
