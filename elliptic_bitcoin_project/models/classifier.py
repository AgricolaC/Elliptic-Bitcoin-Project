import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import Config

def build_loss(cfg: Config, class_weights: torch.Tensor) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights)

class SGCHead(nn.Module):
    """MLP head if cfg.use_mlp_head else a single Linear."""
    def __init__(self, in_dim: int, cfg: Config, n_classes: int = 2):
        super().__init__()
        if cfg.use_mlp_head:
            p = cfg.mlp_dropout
            layers = []
            curr_dim = in_dim
            for h in cfg.mlp_hidden:
                layers.extend([
                    nn.Linear(curr_dim, h),
                    nn.ReLU(),
                    nn.Dropout(p)
                ])
                curr_dim = h
            layers.append(nn.Linear(curr_dim, n_classes))
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(in_dim, n_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


