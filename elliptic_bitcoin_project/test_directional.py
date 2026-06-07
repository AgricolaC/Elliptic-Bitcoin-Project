import torch
import pytest
from models.layers import sgc_propagate

def test_directional_propagation_shape():
    """
    Directional mode must produce (N, d*(1+3*K)) output for K hops.
    Separate from existing fixtures which only test the symmetric path.
    """
    N, d, K = 50, 8, 2
    x = torch.randn(N, d)
    # Small ring graph so no isolated nodes
    src = torch.arange(N)
    dst = (torch.arange(N) + 1) % N
    edge_index = torch.stack([src, dst])

    out = sgc_propagate(x, edge_index, k=K, multiscale=True, use_directional=True)
    expected_dim = d * (1 + 3 * K)   # = 8 * 7 = 56
    assert out.shape == (N, expected_dim), \
        f"Directional shape {out.shape} != ({N}, {expected_dim})"

def test_directional_channels_differ_on_dag():
    """
    On a directed (non-symmetric) graph, S_out·X and S_in·X must differ.
    If they're equal, directionality is being discarded somewhere.
    """
    N, d = 10, 4
    x = torch.randn(N, d)
    # Strictly directed chain: 0→1→2→...→9 (no reverse edges)
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([src, dst])

    out = sgc_propagate(x, edge_index, k=1, multiscale=True, use_directional=True)
    # Columns d..2d are S_sym·X, 2d..3d are S_out·X, 3d..4d are S_in·X
    s_out_block = out[:, 2*d:3*d]
    s_in_block  = out[:, 3*d:4*d]
    assert not torch.allclose(s_out_block, s_in_block), \
        "S_out and S_in are identical — directionality is not being preserved"
