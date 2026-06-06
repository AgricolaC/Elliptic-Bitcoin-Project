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

def sgc_propagate(x: torch.Tensor, edge_index: torch.Tensor, k: int, multiscale: bool) -> torch.Tensor:
    """Return [X | SX | ... | S^K X] if multiscale else S^K X."""
    n, d = x.shape
    assert edge_index.shape[0] == 2, "edge_index must be [2,E]"
    S = gcn_norm(edge_index, n)
    hops = [x]
    cur = x
    for _ in range(k):
        cur = torch.sparse.mm(S, cur)
        hops.append(cur)
    if multiscale:
        out = torch.cat(hops, dim=1)
        assert out.shape == (n, (k + 1) * d), \
            f"Multiscale shape {out.shape} != {(n,(k+1)*d)}"
    else:
        out = cur
        assert out.shape == (n, d), f"Propagation changed shape: {out.shape}"
    return out
