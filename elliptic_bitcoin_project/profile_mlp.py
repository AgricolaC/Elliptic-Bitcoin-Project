import time
import torch
import torch.nn as nn
from models.classifier import SGCHead
from config import Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_forward_pass(model, x, num_iters=100):
    # Warmup
    for _ in range(10):
        _ = model(x)
        
    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    end_time = time.perf_counter()
    
    return (end_time - start_time) / num_iters * 1000  # Return ms per pass

def main():
    # Input dimension: 166 base + 2 topology = 168
    # K=2 multiscale: 168 * 3 = 504
    in_dim = 504
    n_nodes = 40000  # Approx nodes in a graph
    x = torch.randn(n_nodes, in_dim)
    
    # OLD Configuration (128, 64)
    cfg_old = Config()
    cfg_old.use_mlp_head = True
    cfg_old.mlp_hidden = (128, 64)
    model_old = SGCHead(in_dim, cfg_old)
    
    # NEW Configuration (512, 256, 128)
    cfg_new = Config()
    cfg_new.use_mlp_head = True
    cfg_new.mlp_hidden = (512, 256, 128)
    model_new = SGCHead(in_dim, cfg_new)
    
    params_old = count_parameters(model_old)
    params_new = count_parameters(model_new)
    
    ms_old = benchmark_forward_pass(model_old, x)
    ms_new = benchmark_forward_pass(model_new, x)
    
    print("=== Computational Cost Comparison ===")
    print(f"Graph Size: {n_nodes} nodes, Input Dim: {in_dim}")
    print("\n[Old MLP Head: 128 -> 64]")
    print(f"Total Parameters:  {params_old:,}")
    print(f"Forward Pass Time: {ms_old:.2f} ms")
    
    print("\n[New MLP Head: 512 -> 256 -> 128]")
    print(f"Total Parameters:  {params_new:,}")
    print(f"Forward Pass Time: {ms_new:.2f} ms")
    
    print(f"\nCapacity Increase: {params_new / params_old:.1f}x larger")
    print(f"Absolute Time Difference: +{ms_new - ms_old:.2f} ms per epoch")

if __name__ == "__main__":
    main()
