import torch

def gcn_norm(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops as sparse COO (O(E)).
    Applies graph symmetrization A = max(A, A^T) for the DAG.
    """
    # Symmetrize A = max(A, A^T) by concatenating and taking unique (since values are 1)
    sym_idx = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    loop = torch.arange(n)
    sym_idx = torch.cat([sym_idx, torch.stack([loop, loop])], dim=1)
    
    # Coalesce creates sum by default for duplicates, but we want max(A, A^T) which is just 1.0 for unweighted
    A = torch.sparse_coo_tensor(sym_idx, torch.ones(sym_idx.shape[1]), (n, n)).coalesce()
    idx = A.indices()
    val = (A.values() > 0).float()  # this enforces max(A, A^T) -> 1.0
    
    deg = torch.sparse.sum(torch.sparse_coo_tensor(idx, val, (n, n)), dim=1).to_dense()
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    r, c = idx
    return torch.sparse_coo_tensor(idx, dinv[r] * val * dinv[c], (n, n)).coalesce()

def _row_normalize(edge_index: torch.Tensor, n: int) -> torch.sparse.Tensor:
    """
    Row-normalized adjacency: S = D^{-1} A.
    Each row sums to 1 (or 0 for isolated nodes).
    Used for directional propagation where we want a simple mean over neighbors.
    """
    if edge_index.shape[1] == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (n, n)
        ).coalesce()
    r, c = edge_index
    # Degree = number of outgoing edges per source node
    deg = torch.bincount(r, minlength=n).float()
    dinv = deg.pow(-1.0)
    dinv[torch.isinf(dinv)] = 0.0
    vals = dinv[r]
    return torch.sparse_coo_tensor(edge_index, vals, (n, n)).coalesce()

def sgc_propagate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    k: int,
    multiscale: bool,
    use_directional: bool = False,
) -> torch.Tensor:
    """
    Propagate node features with optional directional channels.

    Standard mode (use_directional=False):
        Returns [X | SX | ... | S^K X] if multiscale else S^K X.
        S = D^{-1/2}(A + A^T + I)D^{-1/2}  (symmetric, existing behavior)

    Directional mode (use_directional=True, multiscale must be True):
        Returns [X | S_sym·X | S_out·X | S_in·X] for K=1, or extends each
        operator independently for K>1.

        S_sym = symmetric operator (undirected, existing)
        S_out = D_out^{-1} A       (forward: aggregate from predecessors)
        S_in  = D_in^{-1} A^T     (backward: aggregate from successors)

        Output shape: (N, d * (1 + 3*K))  vs current (N, d * (K+1))
    """
    n, d = x.shape
    assert edge_index.shape[0] == 2, "edge_index must be [2, E]"

    if not use_directional:
        S = gcn_norm(edge_index, n)
        hops = [x]
        cur = x
        for _ in range(k):
            cur = torch.sparse.mm(S, cur)
            hops.append(cur)
        if multiscale:
            out = torch.cat(hops, dim=1)
            assert out.shape == (n, (k + 1) * d)
        else:
            out = hops[-1]
            assert out.shape == (n, d)
        return out

    assert multiscale, "Directional propagation requires multiscale=True (need all scales)"

    S_sym = gcn_norm(edge_index, n)
    # Forward: A propagates predecessor features to each node
    S_out = _row_normalize(edge_index, n)
    # Backward: A^T propagates successor features to each node
    S_in  = _row_normalize(edge_index.flip(0), n)

    # For each operator, accumulate K hops
    def multi_hop(S, x, k):
        hops = []
        cur = x
        for _ in range(k):
            cur = torch.sparse.mm(S, cur)
            hops.append(cur)
        return hops

    sym_hops = multi_hop(S_sym, x, k)
    out_hops = multi_hop(S_out, x, k)
    in_hops  = multi_hop(S_in,  x, k)

    # Interleave by hop depth: [X, S_sym·X, S_out·X, S_in·X, S_sym²·X, ...]
    channels = [x]
    for i in range(k):
        channels.extend([sym_hops[i], out_hops[i], in_hops[i]])

    out = torch.cat(channels, dim=1)
    expected_dim = d * (1 + 3 * k)
    assert out.shape == (n, expected_dim), \
        f"Directional propagation shape {out.shape} != (n, {expected_dim})"
    return out
