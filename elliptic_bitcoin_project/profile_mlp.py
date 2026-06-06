import time
import torch
import torch.nn as nn
from models.classifier import SGCHead
from config import Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_train_loop(model, x, y, num_epochs=200):
    # Setup standard loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    
    # Warmup
    for _ in range(3):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        
    start_time = time.perf_counter()
    for _ in range(num_epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    ms_per_epoch = (total_time / num_epochs) * 1000
    return total_time, ms_per_epoch

def main():
    # Input dimension: 166 base + 2 topology = 168
    # K=2 multiscale: 168 * 3 = 504
    # Real dataset expands to ~507 due to edge logic if we used degree, but we use 504.
    in_dim = 504
    n_nodes = 40000  # Approx nodes in a graph
    x = torch.randn(n_nodes, in_dim)
    y = torch.randint(0, 2, (n_nodes,))
    
    print(f"=== Training Cost Benchmark ===")
    print(f"Dataset Size: {n_nodes:,} nodes")
    print(f"Input Dimension: {in_dim}")
    print(f"Task: 200 Epochs (Forward + Backward + Adam Step)\n")
    
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
    
    total_old, ms_old = benchmark_train_loop(model_old, x, y)
    total_new, ms_new = benchmark_train_loop(model_new, x, y)
    
    print(f"[Old MLP Head: 128 -> 64]")
    print(f"  Total Parameters: {params_old:,}")
    print(f"  Cost per Epoch:   {ms_old:.2f} ms")
    print(f"  Total 200 Epochs: {total_old:.3f} seconds\n")
    
    print(f"[New MLP Head: 512 -> 256 -> 128]")
    print(f"  Total Parameters: {params_new:,}")
    print(f"  Cost per Epoch:   {ms_new:.2f} ms")
    print(f"  Total 200 Epochs: {total_new:.3f} seconds\n")
    
    print(f"--- Conclusion ---")
    print(f"Capacity Increase:      {params_new / params_old:.1f}x larger")
    print(f"Single Fit Penalty:     +{total_new - total_old:.3f} seconds per 200-epoch fit")

if __name__ == "__main__":
    main()
