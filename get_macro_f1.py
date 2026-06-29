import pandas as pd
import numpy as np

df = pd.read_csv('results/walk_forward_timesteps.csv')

def get_phase(tau):
    if tau <= 42:
        return 'Pre-Shock'
    elif tau == 43:
        return 'Shock'
    else:
        return 'Recovery'

df['Phase'] = df['Tau'].apply(get_phase)

# Filter out Ablations and only keep Baselines and WF Champions
df = df[df['Variation'].str.contains('Baseline|WF Champion|Ablation: IPCA') == True]

grouped = df.groupby(['Variation', 'Phase'])['F1'].mean().unstack()

# Also get Overall Macro F1
overall = df.groupby('Variation')['F1'].mean()
grouped['Overall'] = overall

# Add OOT Macro F1 from sweep_results for comparison (if we want)
sweep = pd.read_csv('results/sweep_results.csv')
sweep_wf = sweep[sweep['Variation'].str.contains('Baseline|WF Champion|Ablation: IPCA') == True]
sweep_wf = sweep_wf[sweep_wf['Seed'] == 42]
sweep_dict = dict(zip(sweep_wf['Variation'], sweep_wf['Static_OOT_Macro_F1']))
grouped['OOT_Macro_F1'] = grouped.index.map(sweep_dict)

print(grouped[['OOT_Macro_F1', 'Overall', 'Pre-Shock', 'Shock', 'Recovery']].round(3).to_string())
