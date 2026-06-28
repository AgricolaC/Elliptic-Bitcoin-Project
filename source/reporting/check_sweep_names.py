import pandas as pd
import re

def main():
    # Load data
    try:
        results_df = pd.read_csv('results/final_aggregated_results.csv')
        timesteps_df = pd.read_csv('results/final_aggregated_timesteps.csv')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    csv_names = set(results_df['Sweep'].dropna().unique()).union(
        set(timesteps_df['Sweep'].dropna().unique())
    )
    
    # Read build_presentation.py
    try:
        with open('source/reporting/build_presentation.py', 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading build_presentation.py: {e}")
        return

    # Extract all sweep names used in build_presentation.py
    # Look for get_scalar('...', ...) or ('...', '...', ...) or == '...'
    # Specifically looking for F1:, F3, F4:, Diagnostic:, Baseline:, Best WF:, Ablation:
    pattern = re.compile(r"'(F1:|F3[a-z]?:|F4:|Diagnostic:|Baseline:|Best WF:|Ablation:|Grid:|Sweep \d:) [^']+'|'(Diagnostic:|F3[a-z]?:) [^']+'")
    
    # A broader approach: look for all string literals that match the old or new prefixes
    # Or just extract what's in the design document:
    names_to_check = [
        'Diagnostic: sklearn LR',
        'F3d: GCN reference [2-layer]',
        'F3a: Base XGBoost (clean)',
        'F3b: Random Forest (clean)',
        'F1: SGC+MLP WF K=2 [Dir=F; Topo=None]',
        'F1: SGC+MLP WF K=5 [Dir=F; Topo=None; PCA]',
        'F4: SGC+MLP decay λ=0.05',
        'F4: SGC+MLP decay λ=0.25',
        'F4: SGC+MLP decay λ=0.5',
    ]

    print("--- Sweep Name Hit/Miss Table ---")
    for name in names_to_check:
        if name in csv_names:
            print(f"✅ FOUND: '{name}'")
        else:
            print(f"❌ MISSING: '{name}'")
            # Find closest match
            closest = [c for c in csv_names if all(word in c for word in name.split() if len(word) > 2)]
            if closest:
                print(f"   -> Closest matches: {closest}")
            else:
                # print all CSV names for manual check
                pass
                
    print("\n--- SGC Decay Candidates ---")
    # For each lambda, print candidates and their WF_Pooled_F1_mean
    for lam in ['0.05', '0.25', '0.5']:
        print(f"Lambda = {lam}:")
        candidates = results_df[results_df['Sweep'].str.contains(f"Decay λ={lam}", na=False)]
        for _, row in candidates.iterrows():
            print(f"  {row['Sweep']}: {row.get('WF_Pooled_F1_mean', 'N/A')}")
            
if __name__ == '__main__':
    main()
