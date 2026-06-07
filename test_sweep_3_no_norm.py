import os
import sys

# Temporarily patch models/layers.py to remove the per-hop std normalization
import models.layers
with open("elliptic_bitcoin_project/models/layers.py", "r") as f:
    content = f.read()

content = content.replace("""        if multiscale:
            normalized = [hops[0]]
            for h in hops[1:]:
                std = h.std(dim=0, keepdim=True).clamp(min=1e-6)
                normalized.append(h / std)
            out = torch.cat(normalized, dim=1)""", """        if multiscale:
            out = torch.cat(hops, dim=1)""")

with open("elliptic_bitcoin_project/models/layers.py", "w") as f:
    f.write(content)

# Run Sweep 3
import pandas as pd
from config import Config
from data.load_dataset import download_and_load_data
from run_sweeps import run_static_only_sweep, _RESULT_KEYS

df, df_edge, _, feature_cols = download_and_load_data()
cfg = Config(use_mlp_head=True, use_multiscale_prop=True, use_graph_structural=False)
res = run_static_only_sweep("Sweep 3: No Norm", cfg, df, df_edge, feature_cols)
print(f"Sweep 3 Raw Concatenation F1: {res['Static OOT F1']}")

# Restore models/layers.py
with open("elliptic_bitcoin_project/models/layers.py", "w") as f:
    f.write(content.replace("""        if multiscale:
            out = torch.cat(hops, dim=1)""", """        if multiscale:
            normalized = [hops[0]]
            for h in hops[1:]:
                std = h.std(dim=0, keepdim=True).clamp(min=1e-6)
                normalized.append(h / std)
            out = torch.cat(normalized, dim=1)"""))
