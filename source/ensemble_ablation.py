import os
import sys
import torch
import warnings

# Ensure working directory is correctly set
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.build_graph import EllipticDataModule
from evaluation.validation import walk_forward_validation
from evaluation.ablation_validation import evaluate_decay_wf
from data.load_dataset import download_and_load_data

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dummy_make_result(**kwargs):
    return kwargs

def main():
    print("Loading raw dataset...")
    df, df_edge, _, feature_cols = download_and_load_data()

    print("Initializing Tabular DataModule (XGBoost)...")
    cfg_xgb = Config()
    cfg_xgb.use_multiscale_prop = False
    cfg_xgb.use_directional_prop = False
    cfg_xgb.use_graph_structural = False
    cfg_xgb.sgc_k = 1
    
    dm_xgb = EllipticDataModule(df, df_edge, feature_cols, cfg_xgb)
    dm_xgb.setup()

    print("\nInitializing Graph DataModule (K=3, Dir=T, Topo=late, PCA)...")
    cfg_sgc = Config()
    cfg_sgc.sgc_k = 3
    cfg_sgc.use_directional_prop = True
    cfg_sgc.use_graph_structural = True
    cfg_sgc.topo_injection_mode = "late"
    cfg_sgc.use_ipca = False
    cfg_sgc.use_pca = True
    
    dm_sgc = EllipticDataModule(df, df_edge, feature_cols, cfg_sgc)
    dm_sgc.setup()

    w_name = "Ensemble: XGB(0.7) + 3 T late PCA(0.3)"
    print(f"\nEvaluating {w_name}...")
    
    walk_forward_validation(
        dm=dm_sgc,
        cfg=cfg_sgc,
        device=DEVICE,
        sweep_name=w_name,
        return_records=False,
        window=None,
        eval_steps=cfg_sgc.test_steps,
        xgb_dm=dm_xgb,
        xgb_weight=0.7
    )
    
    print("\nInitializing Graph DataModule (K=3, Dir=T, Topo=early, Base)...")
    cfg_base = Config()
    cfg_base.sgc_k = 3
    cfg_base.use_directional_prop = True
    cfg_base.use_graph_structural = True
    cfg_base.topo_injection_mode = "early"
    cfg_base.use_pca = False
    cfg_base.use_ipca = False
    
    dm_base = EllipticDataModule(df, df_edge, feature_cols, cfg_base)
    dm_base.setup()

    w_name_2 = "Grid: K=3, Dir=T, Topo=early (Var Base) WF"
    print(f"\nEvaluating {w_name_2}...")
    
    walk_forward_validation(
        dm=dm_base,
        cfg=cfg_base,
        device=DEVICE,
        sweep_name=w_name_2,
        return_records=False,
        window=None,
        eval_steps=cfg_base.test_steps,
        xgb_dm=None
    )

    lambda_decay = 0.25
    print(f"\nEvaluating Ablation: Decay λ={lambda_decay} on 3 T early Base...")
    evaluate_decay_wf(dm_base, cfg_base, lambda_decay, "Ablation: Decay λ=0.25 on 3 T early Base", dummy_make_result)

if __name__ == "__main__":
    main()
