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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

@dataclass
class Config:
    train_steps: range = range(1, 27)
    val_steps:   range = range(27, 35)
    test_steps:  range = range(35, 50)
    disruption_step: int = 43

    # Original ablatable mechanisms
    use_graph_structural: bool = True
    class_weighted: bool = True

    # Architectural modifications
    use_multiscale_prop: bool = True   # [X | S^1 X | ... | S^K X] vs S^K X only
    use_mlp_head: bool = True          # 3-layer MLP head vs single Linear
    use_xgb_head: bool = False         # XGBoost head
    use_directional_prop: bool = False # Directional DAG channels
    topo_injection_mode: str = 'late'  # 'early' (smoothed) vs 'late' (anchored)
    
    # Temporal (D2)
    use_lstm_head: bool = False        # LSTM temporal conditioning
    lstm_hidden: int = 64              # LSTM hidden size
    use_ema_weights: bool = False      # Exponential Moving Average of weights

    # SGC Hyperparameters
    sgc_k: int = 2
    sgc_epochs: int = 200
    wf_epochs: int = 100                # reduction in epoch for walk-forward loop
    sgc_lr: float = 0.01
    sgc_weight_decay: float = 5e-4
    sgc_l1_lambda: float = 1e-4        # ElasticNet L1 penalty for feature selection
    mlp_hidden: tuple = (128, 64)
    mlp_dropout: float = 0.3
    use_layernorm: bool = False
    use_residual: bool = False
    activation: str = 'relu'
    
    # Feature Selection & Dim. Reduction
    use_pca: bool = False
    use_ipca: bool = False             # Walk-forward Incremental PCA
    pca_variance: float = 0.99         # % of variance to retain when use_pca=True

    # Elliptic dataset: first 93 features are local (tx-level), last 72 are
    # pre-aggregated neighbor statistics. When True, SGC propagation runs only
    # on the local block; the agg block bypasses SGC and is concatenated after.
    use_local_only_prop: bool = False
    n_local_features: int = 93

    # Seed
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        all_splits = [
            ("train", self.train_steps),
            ("val",   self.val_steps),
            ("test",  self.test_steps),
        ]
        for i, (n1, s1) in enumerate(all_splits):
            for n2, s2 in all_splits[i+1:]:
                assert set(s1).isdisjoint(s2), \
                    f"Temporal leakage: {n1} and {n2} time steps overlap."
