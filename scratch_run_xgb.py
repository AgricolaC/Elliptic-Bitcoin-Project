import sys, os
import joblib

sys.path.append('source')
from config import Config, set_global_seeds, DEVICE, OUTPUT_DIR
from data.load_dataset import download_and_load_data
from data.build_graph import EllipticDataModule
from sweep import walk_forward_baseline
from xgboost import XGBClassifier

def main():
    set_global_seeds(42)
    df, df_edge, _, feature_cols = download_and_load_data()
    
    cfg = Config()
    dm = EllipticDataModule(df, df_edge, feature_cols, cfg)
    dm.setup()
    
    print("Running Baseline XGBoost (166) Walk-Forward...")
    
    # We pass window=None, use_temporal=False
    # scale_pos_weight is dynamically updated per timestep inside the function!
    walk_forward_baseline(
        dm, cfg, XGBClassifier, sweep_name="Baseline: XGBoost (166)", window=None,
        n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=1.0, eval_metric="aucpr", random_state=cfg.seed, n_jobs=1
    )
    print("Appended successfully.")

if __name__ == "__main__":
    main()
