import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import Config

class FocalLoss(nn.Module):
    """Multiclass focal loss. gamma=0 reduces exactly to weighted CE."""
    def __init__(self, alpha: torch.Tensor, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        at = self.alpha.gather(0, targets)
        return -(at * (1 - pt).pow(self.gamma) * logpt).mean()

def build_loss(cfg: Config, class_weights: torch.Tensor) -> nn.Module:
    if cfg.use_focal_loss:
        return FocalLoss(class_weights, cfg.focal_gamma)
    return nn.CrossEntropyLoss(weight=class_weights)

class SGCHead(nn.Module):
    """MLP head if cfg.use_mlp_head else a single Linear."""
    def __init__(self, in_dim: int, cfg: Config, n_classes: int = 2):
        super().__init__()
        if cfg.use_mlp_head:
            h1, h2 = cfg.mlp_hidden
            p = cfg.mlp_dropout
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(p),
                nn.Linear(h1, h2),     nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(p),
                nn.Linear(h2, n_classes),
            )
        else:
            self.net = nn.Linear(in_dim, n_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


