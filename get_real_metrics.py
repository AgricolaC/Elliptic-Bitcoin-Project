import pandas as pd

df = pd.read_csv("results/final_aggregated_results.csv")
print(df[['Sweep', 'Variation', 'Static_OOT_Macro_PRAUC_mean']].dropna())
