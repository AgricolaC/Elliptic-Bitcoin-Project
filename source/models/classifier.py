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
        self.use_mlp_head = cfg.use_mlp_head
        self.use_residual = getattr(cfg, "use_residual", False) and cfg.use_mlp_head
        
        if self.use_mlp_head:
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
            
            self.hidden_net = nn.Sequential(*layers)
            self.classifier = nn.Linear(curr_dim, n_classes)
            
            if self.use_residual:
                if in_dim == curr_dim:
                    self.shortcut = nn.Identity()
                else:
                    self.shortcut = nn.Linear(in_dim, curr_dim)
        else:
            self._net = nn.Sequential(
                nn.Linear(in_dim, n_classes)
            )
            
    @property
    def net(self) -> nn.Sequential:
        return self.hidden_net if self.use_mlp_head else self._net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mlp_head:
            h = self.hidden_net(x)
            if self.use_residual:
                h = F.relu(h + self.shortcut(x))
            return self.classifier(h)
        else:
            return self._net(x)

