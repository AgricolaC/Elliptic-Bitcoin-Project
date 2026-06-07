import sys
import os
import time
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class SGCHeadOld(nn.Module):
    def __init__(self, in_dim: int, cfg: Config, n_classes: int = 2):
        super().__init__()
        if cfg.use_mlp_head:
            p = cfg.mlp_dropout
            layers = []
            curr_dim = in_dim
            for h in cfg.mlp_hidden:
                layers.extend([
                    nn.Linear(curr_dim, h),
                    nn.BatchNorm1d(h),
                    nn.GELU(),
                    nn.Dropout(p)
                ])
                curr_dim = h
            layers.append(nn.Linear(curr_dim, n_classes))
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(in_dim, n_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

from models.classifier import SGCHead as SGCHeadNew

def run():
    torch.manual_seed(42)

    # Simulate the actual input shape: (N_labeled_nodes, sgc_input_dim)
    # sgc_input_dim for K=2 multiscale = 168 * 3 = 504 (approx)
    N, D = 3000, 504
    X = torch.randn(N, D)
    y = torch.randint(0, 2, (N,))
    weights = torch.tensor([0.5, 10.0])  # class-weighted

    cfg = Config(use_mlp_head=True)

    model_old = SGCHeadOld(D, cfg)
    model_new = SGCHeadNew(D, cfg)

    # Time 50 forward+backward passes
    def time_head(model, n_passes=50):
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        model.train()
        t0 = time.perf_counter()
        for _ in range(n_passes):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        return (time.perf_counter() - t0) / n_passes * 1000  # ms/pass

    print("Running micro-benchmark...")
    time_old = time_head(model_old)
    time_new = time_head(model_new)
    
    print(f"Old Architecture (GELU + BatchNorm): {time_old:.2f} ms / epoch")
    print(f"New Architecture (ReLU, No BatchNorm): {time_new:.2f} ms / epoch")
    
    speedup = (time_old - time_new) / time_old * 100
    print(f"Speedup: {speedup:.1f}%")

if __name__ == "__main__":
    run()
