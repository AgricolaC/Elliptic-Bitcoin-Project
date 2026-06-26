import pandas as pd
import numpy as np

files = [
    'results/eda_degree.csv',
    'results/eda_pagerank.csv',
    'results/eda_pca.csv',
    'results/eda_tsne.csv'
]

def q01(x): return x.quantile(0.01)
def q05(x): return x.quantile(0.05)
def q25(x): return x.quantile(0.25)
def q50(x): return x.quantile(0.50)
def q75(x): return x.quantile(0.75)
def q95(x): return x.quantile(0.95)
def q99(x): return x.quantile(0.99)
def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
def mad(x): return (x - x.median()).abs().median()
def skewness(x): return x.skew()
def kurtosis(x): return x.kurt()

# Give descriptive names for output
q01.__name__ = '1%'
q05.__name__ = '5%'
q25.__name__ = '25%'
q50.__name__ = '50%'
q75.__name__ = '75%'
q95.__name__ = '95%'
q99.__name__ = '99%'
iqr.__name__ = 'iqr'
mad.__name__ = 'mad'
skewness.__name__ = 'skewness'
kurtosis.__name__ = 'kurtosis'

aggs = [
    'count', 'mean', 'std', 'min', 
    q01, q05, q25, q50, q75, q95, q99, 'max',
    iqr, mad, skewness, kurtosis
]

for f in files:
    df = pd.read_csv(f)
    if 'tau' in df.columns:
        df = df.drop(columns=['tau'])
        
    stats = df.groupby('label').agg(aggs)
    
    # Flatten multi-index columns
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns.values]
    
    # Calculate correlations between features if there are exactly 2 features (like pca1/pca2)
    features = [c for c in df.columns if c != 'label']
    if len(features) == 2:
        f1, f2 = features[0], features[1]
        corr_name = f"{f1}_{f2}_correlation"
        corr_series = df.groupby('label').apply(lambda g: g[f1].corr(g[f2]), include_groups=False)
        stats[corr_name] = corr_series
        
    out_file = f.replace('.csv', '_stats.csv')
    stats.to_csv(out_file)
    print(f"Saved advanced stats to {out_file}")

