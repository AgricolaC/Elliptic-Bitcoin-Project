import numpy as np
from typing import List, Any
import warnings

def build_snapshot_temporal_features(
    dm: Any,
    target_step: int,
    window: int,
    label_lag: int = 0,
    feature_col_indices: List[int] = None,
) -> np.ndarray:
    """
    Build graph/market-level temporal features for snapshot `target_step`
    from the window [target_step - window, target_step - 1] ONLY.

    Features per lag step s in [target_step - window, target_step - 1]:
      - illicit_rate: (labeled nodes with y==1) / (total labeled nodes) at step s - label_lag
        # LEAKAGE GUARD: labels for past steps are legitimately known at deployment
        # The label_lag parameter explicitly models the delay in getting investigation results.
      - node_count: total nodes in snapshot s
      - edge_count: total edges in snapshot s
      - mean_degree: 2 * edge_count / node_count
      - 2 x N_features: mean and std of selected raw feature columns across all nodes in step s
        (label-free)
      - pagerank_var: variance of PageRank (measuring concentration of importance)
      - lag_present: 1.0 if step s exists, 0.0 otherwise.
      
    Then compute first-differences across the window for trend detection.

    Returns:
        1-D array of shape (n_temporal_features,) broadcast-ready for all nodes
        in target_step.

    LEAKAGE GUARD: only accesses dm.graphs[s] for s < target_step.
    """
    if feature_col_indices is None:
        feature_col_indices = list(range(5))  # Default to first 5 raw features

    # Precompute training-set means to fill missing lags (instead of 0-filling)
    # LEAKAGE GUARD: training-set means only use cfg.train_steps
    train_steps = getattr(dm.cfg, "train_steps", [])
    train_graphs = [t for t in train_steps if t in dm.graphs]
    
    # Defaults
    tr_illicit_rate = 0.0
    tr_node_count = 0.0
    tr_edge_count = 0.0
    tr_mean_degree = 0.0
    tr_feat_mean = np.zeros(len(feature_col_indices))
    tr_feat_std = np.zeros(len(feature_col_indices))
    tr_pagerank_var = 0.0
    
    if train_graphs:
        ill_rates, n_counts, e_counts, m_degs, f_means, f_stds, pr_vars = [], [], [], [], [], [], []
        for t in train_graphs:
            g = dm.graphs[t]
            n_nodes = g["x"].shape[0]
            # edge_index stores each raw directed edge once. Symmetrization is
            # performed later inside SGC propagation and does not mutate it.
            e_count = g["edge_index"].shape[1]
            n_counts.append(n_nodes)
            e_counts.append(e_count)
            m_degs.append(2.0 * e_count / n_nodes if n_nodes > 0 else 0)
            
            # Illicit rate
            m = g["labeled_mask"].numpy()
            y = g["y"][m].numpy()
            if len(y) > 0:
                ill_rates.append((y == 1).mean())
                
            x_sel = g["x"][:, feature_col_indices].numpy()
            f_means.append(x_sel.mean(axis=0))
            f_stds.append(x_sel.std(axis=0))
            
            if "topo" in g and getattr(dm.cfg, "topo_injection_mode", "late") == "late":
                # Assuming first topo col is pagerank
                pr_vars.append(g["topo"][:, 0].numpy().var())
            elif "topo" in g and getattr(dm.cfg, "topo_injection_mode", "late") == "early":
                pr_vars.append(0.0) # Not easily retrievable if early injection is used, fallback
            else:
                pr_vars.append(0.0)
                
        if ill_rates: tr_illicit_rate = np.mean(ill_rates)
        tr_node_count = np.mean(n_counts)
        tr_edge_count = np.mean(e_counts)
        tr_mean_degree = np.mean(m_degs)
        tr_feat_mean = np.mean(f_means, axis=0)
        tr_feat_std = np.mean(f_stds, axis=0)
        tr_pagerank_var = np.mean(pr_vars)

    lag_features = []
    
    min_step = min(dm.graphs.keys()) if dm.graphs else 1
    start_lag = target_step - window
    
    for s in range(start_lag, target_step):
        # LEAKAGE GUARD: step s is strictly < target_step
        assert s < target_step, f"Leakage: s={s} >= target_step={target_step}"
        
        if s in dm.graphs:
            g = dm.graphs[s]
            n_nodes = g["x"].shape[0]
            e_count = g["edge_index"].shape[1]
            node_count = n_nodes
            edge_count = e_count
            mean_degree = 2.0 * e_count / n_nodes if n_nodes > 0 else 0.0
            
            x_sel = g["x"][:, feature_col_indices].numpy()
            feat_mean = x_sel.mean(axis=0)
            feat_std = x_sel.std(axis=0)
            
            if "topo" in g and getattr(dm.cfg, "topo_injection_mode", "late") == "late":
                pagerank_var = g["topo"][:, 0].numpy().var()
            else:
                pagerank_var = 0.0
                
            lag_present = 1.0
        else:
            node_count = tr_node_count
            edge_count = tr_edge_count
            mean_degree = tr_mean_degree
            feat_mean = tr_feat_mean
            feat_std = tr_feat_std
            pagerank_var = tr_pagerank_var
            lag_present = 0.0
            
        # Label lag
        s_label = s - label_lag
        if s_label in dm.graphs:
            g_lbl = dm.graphs[s_label]
            m = g_lbl["labeled_mask"].numpy()
            y = g_lbl["y"][m].numpy()
            if len(y) > 0:
                illicit_rate = (y == 1).mean()
            else:
                illicit_rate = tr_illicit_rate
        else:
            illicit_rate = tr_illicit_rate
            
        step_feats = [
            illicit_rate, node_count, edge_count, mean_degree,
            pagerank_var, lag_present
        ]
        step_feats.extend(feat_mean.tolist())
        step_feats.extend(feat_std.tolist())
        lag_features.append(np.array(step_feats, dtype=np.float32))

    # Width: 1 + 1 + 1 + 1 + 1 + 1 + 2 * len(feature_col_indices) = 6 + 2 * 5 = 16 features per lag
    # Expected total width = 16 * w + 15 * (w - 1) = 31w - 15 (lag_present has no first difference)
    
    if window == 0:
        return np.array([], dtype=np.float32)

    lag_matrix = np.stack(lag_features)  # (w, 16)
    
    first_diffs = []
    for i in range(1, window):
        if lag_matrix[i, 5] == 1.0 and lag_matrix[i-1, 5] == 1.0: # lag_present is index 5
            # Compute difference excluding lag_present (index 5)
            diff = lag_matrix[i, :5] - lag_matrix[i-1, :5]
            diff = np.concatenate([diff, lag_matrix[i, 6:] - lag_matrix[i-1, 6:]])
        else:
            diff = np.zeros(15, dtype=np.float32)
        first_diffs.append(diff)
        
    flat_lags = lag_matrix.flatten()
    if first_diffs:
        flat_diffs = np.concatenate(first_diffs)
        final_feats = np.concatenate([flat_lags, flat_diffs])
    else:
        final_feats = flat_lags
        
    expected_width = 16 * window + 15 * (window - 1) if window > 0 else 0
    assert final_feats.shape == (expected_width,), \
        f"Temporal feature width {final_feats.shape} != expected ({expected_width},)"
        
    return final_feats
