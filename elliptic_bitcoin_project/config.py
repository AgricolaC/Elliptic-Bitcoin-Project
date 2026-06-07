import random
import numpy as np
import torch
from dataclasses import dataclass

import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Enforce deterministic behavior across all RNG sources (axiom-defend M1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    train_steps: range = range(1, 35)
    test_steps:  range = range(35, 50)
    disruption_step: int = 43

    # --- original ablatable mechanisms ---
    use_graph_structural: bool = True
    class_weighted: bool = True

    # --- architectural modifications ---
    use_multiscale_prop: bool = True   # [X | S^1 X | ... | S^K X] vs S^K X only
    use_mlp_head: bool = True          # 3-layer MLP head vs single Linear

    # --- exposed magnitudes ---
    sgc_k: int = 2
    sgc_epochs: int = 200
    wf_epochs: int = 70                # optimization for walk-forward loop
    sgc_lr: float = 0.01
    sgc_weight_decay: float = 5e-4
    focal_gamma: float = 2.0           # 0.0 == weighted CE
    mlp_hidden: tuple = (128, 64)
    mlp_dropout: float = 0.3

    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        assert set(self.train_steps).isdisjoint(self.test_steps), \
            "Temporal leakage: train and test time steps overlap."
