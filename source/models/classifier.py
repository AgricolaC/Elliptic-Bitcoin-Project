import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import Config

def build_loss(cfg: Config, class_weights: torch.Tensor) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights)

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, p: float, use_ln: bool, act_type: str, use_residual: bool):
        super().__init__()
        self.use_residual = use_residual
        self.lin = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()
        self.act = F.silu if act_type.lower() == 'silu' else F.relu
        self.drop = nn.Dropout(p)
        
        if self.use_residual:
            self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
            
    def forward(self, x):
        h = self.lin(x)
        h = self.ln(h)
        if self.use_residual:
            h = h + self.shortcut(x)
        h = self.act(h)
        return self.drop(h)

class SGCHead(nn.Module):
    """Deep Residual MLP head or a single Linear."""
    def __init__(self, in_dim: int, cfg: Config, n_classes: int = 2):
        super().__init__()
        self.use_mlp_head = cfg.use_mlp_head
        
        if self.use_mlp_head:
            p = getattr(cfg, "mlp_dropout", 0.3)
            use_ln = getattr(cfg, "use_layernorm", False)
            act_type = getattr(cfg, "activation", "relu")
            use_res = getattr(cfg, "use_residual", False)
            
            blocks = []
            curr_dim = in_dim
            for h in cfg.mlp_hidden:
                blocks.append(MLPBlock(curr_dim, h, p, use_ln, act_type, use_res))
                curr_dim = h
            
            self.hidden_net = nn.Sequential(*blocks)
            self.classifier = nn.Linear(curr_dim, n_classes)
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
            return self.classifier(h)
        else:
            return self._net(x)

