def test_propagated_hops_have_unit_std():
    """
    After per-hop normalization, each propagated block (hops 1..K) should
    have per-feature std ≈ 1.0. Hop 0 (X) should be unchanged.
    """
    import torch
    from models.layers import sgc_propagate

    torch.manual_seed(42)
    N, d, K = 200, 8, 2
    x = torch.randn(N, d)  # already ~unit variance
    src = torch.arange(N)
    dst = (torch.arange(N) + 1) % N
    edge_index = torch.stack([src, dst])

    out = sgc_propagate(x, edge_index, k=K, multiscale=True, use_directional=False)
    # out shape: (N, d*(K+1)) = (200, 24)
    for hop_idx in range(1, K + 1):
        block = out[:, hop_idx * d : (hop_idx + 1) * d]
        stds = block.std(dim=0)
        assert (stds - 1.0).abs().max() < 0.1, \
            f"Hop {hop_idx} std not ≈ 1.0: {stds}"

    # X block should be unchanged
    x_block = out[:, :d]
    assert torch.allclose(x_block, x), "Hop 0 (X) was modified — should be untouched"
