"""
Build a presentation-style Jupyter notebook for the Elliptic Bitcoin project.

The notebook is generated from two evidence sources:
  1. results/*.csv and results/deep_res_mlp_results/** for experimental numbers
  2. source/** and source/reporting/results/*.md for implementation and analysis context

It intentionally embeds generated PNG assets in Markdown cells, so the notebook
opens as a complete presentation even without executing cells.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sweep_parser import add_parsed_columns, select


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SOURCE = ROOT / "source"
OUT_DIR = ROOT / "presentation"
ASSETS = OUT_DIR / "assets"
NOTEBOOK = OUT_DIR / "elliptic_bitcoin_math_presentation.ipynb"


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for stale_asset in ASSETS.glob("*.png"):
        stale_asset.unlink()


def fmt(x, digits: int = 3) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except TypeError:
        pass
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def markdown_table(headers: list[str], rows: list[list], digits: int = 3) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(v, digits) for v in row) + " |")
    return "\n".join(out)


def metric_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def savefig(name: str) -> str:
    path = ASSETS / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"assets/{name}"




def plot_sgc_oversmoothing(df_fa: pd.DataFrame) -> str:
    df_fa = add_parsed_columns(df_fa.copy())

    # Grid rows only — explicit K=/Dir=/Topo= strings
    grid = df_fa[
        select(df_fa, family_tag='Grid')
        & df_fa['_K'].notna()
    ].copy()
    grid['K'] = grid['_K'].astype(int)
    grid['Var'] = grid['_variation'].fillna(grid['Variation'])

    # Best PRAUC per (K, Var) combination
    pivot = (
        grid
        .groupby(['K', 'Var'])['Static_OOT_Macro_PRAUC_mean']
        .max()
        .unstack('Var')
    )

    k_vals = [1, 2, 3]
    width = 0.35
    base_vals = [pivot.get('Base', pd.Series()).get(k, 0) for k in k_vals]
    pca_vals  = [pivot.get('PCA',  pd.Series()).get(k, 0) for k in k_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = list(range(len(k_vals)))
    ax.bar([x - width/2 for x in xs], base_vals, width,
           label='Raw (Base)', color='#e74c3c', alpha=0.85)
    ax.bar([x + width/2 for x in xs], pca_vals,  width,
           label='PCA',        color='#3498db', alpha=0.85)
    ax.set_xlabel('Neighborhood Depth K')
    ax.set_ylabel('Best OOT Macro PR-AUC  [primary metric]')
    ax.set_title(
        'PCA as Oversmoothing Regularizer\n'
        'K=3 Raw → collapse; K=3 PCA → best graph-model OOT score'
    )
    ax.set_xticks(xs); ax.set_xticklabels(['K=1', 'K=2', 'K=3'])
    ax.legend()
    return savefig("13_sgc_oversmoothing.png")

def plot_cost_vs_performance(sweep_df: pd.DataFrame) -> str:
    df_sr = add_parsed_columns(sweep_df.copy())
    agg = (
        df_sr
        .dropna(subset=['Static_Time_s', 'Static_OOT_Macro_PRAUC'])
        .groupby('_family')
        .agg(
            time_mean=('Static_Time_s', 'mean'),
            prauc_mean=('Static_OOT_Macro_PRAUC', 'mean'),
            prauc_std=('Static_OOT_Macro_PRAUC', 'std'),
        )
        .reset_index()
        .rename(columns={'_family': 'family'})
        .dropna(subset=['prauc_mean'])
    )
    agg = agg[agg['family'] != 'IsolationForest']

    palette = {
        'XGBoost': '#e74c3c', 'RandomForest': '#e67e22',
        'SGC+MLP': '#3498db', 'SGC': '#9b59b6',
        'LogisticRegression': '#1abc9c', 'GCN': '#34495e',
    }
    display_names = {
        'LogisticRegression': 'Logistic Reg.', 'GCN': 'PyG GCN',
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for _, row in agg.iterrows():
        color = palette.get(row['family'], '#7f8c8d')
        ax.scatter(row['time_mean'], row['prauc_mean'],
                   color=color, s=180, zorder=5)
        name = display_names.get(row['family'], row['family'])
        ax.annotate(name, (row['time_mean'], row['prauc_mean']),
                    textcoords='offset points', xytext=(8, 4), fontsize=10)
        if pd.notna(row['prauc_std']) and row['prauc_std'] > 0:
            ax.errorbar(row['time_mean'], row['prauc_mean'],
                        yerr=row['prauc_std'], fmt='none',
                        color='grey', capsize=4, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Training Time (seconds, log scale)')
    ax.set_ylabel('OOT Macro PR-AUC  [primary metric]')
    ax.set_title('Computational Cost vs. OOT Performance\n'
                 'Error bars = ±1 std across 3 seeds (SGC/SGC+MLP)')
    ax.grid(True, alpha=0.3)
    return savefig("12_cost_vs_perf.png")

def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "snapshot": read_csv("snapshot_topology.csv"),
        "drift": read_csv("eda_drift.csv"),
        "separability": read_csv("label_separability.csv"),
        "homophily": read_csv("eda_homophily.csv"),
        "degree": read_csv("eda_degree.csv"),
        "degree_stats": read_csv("eda_degree_stats.csv"),
        "sweep": read_csv("sweep_results.csv"),
        "pca": read_csv("eda_pca.csv"),
        "tsne": read_csv("eda_tsne.csv"),
        "intrinsic": read_csv("eda_grid_intrinsic_dim.csv"),
        "final": read_csv("final_aggregated_results.csv"),
        "timesteps": read_csv("final_aggregated_timesteps.csv"),
        "falsification": read_csv("falsification_log.csv"),
    }

    for phase in "ABCD":
        data[f"phase{phase}"] = pd.read_csv(
            RESULTS / "deep_res_mlp_results" / f"sweep_phase{phase}" / f"phase{phase}_aggregated.csv"
        )

    for run in range(4):
        run_dir = RESULTS / "deep_res_mlp_results" / f"run{run}"
        csvs = sorted(run_dir.glob("deep_res_mlp_results_*.csv"))
        if csvs:
            data[f"run{run}"] = pd.read_csv(csvs[-1])

    return data


def copy_existing_figures() -> dict[str, str]:
    copied = {}
    for src in (RESULTS / "figures").glob("*.png"):
        dst = ASSETS / src.name
        shutil.copy2(src, dst)
        copied[src.stem] = f"assets/{src.name}"
    return copied


def plot_prior_shift(snapshot: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    ax1.plot(snapshot["Tau"], snapshot["N_licit"], label="Licit", color="#4C72B0", lw=2)
    ax1.plot(snapshot["Tau"], snapshot["N_unknown"], label="Unknown", color="#999999", lw=2)
    ax1.plot(snapshot["Tau"], snapshot["N_illicit"], label="Illicit", color="#C44E52", lw=2.4)
    ax1.set_yscale("log")
    ax1.set_xlabel("time step $\\tau$")
    ax1.set_ylabel("node count, log scale")
    ax1.axvline(43, ls="--", color="black", lw=1.8, label="$\\tau=43$ shock")

    ax2 = ax1.twinx()
    ax2.plot(snapshot["Tau"], snapshot["Illicit_Rate"], color="#55A868", lw=2.2, label="Illicit rate")
    ax2.set_ylabel("illicit rate among labeled nodes")
    ax2.grid(False)
    ax1.set_title("The shock is a prior-probability shift: illicit volume collapses at $\\tau=43$")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2)
    return savefig("01_prior_shift.png")


def plot_graph_stability(snapshot: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True)
    axes[0].plot(snapshot["Tau"], snapshot["Mean_Degree"], color="#4C72B0", lw=2.4)
    axes[0].axvline(43, ls="--", color="black")
    axes[0].set_ylabel("mean degree")
    axes[0].set_title("Macroscopic graph structure remains stable through the shock")

    axes[1].plot(snapshot["Tau"], snapshot["Graph_Density"], color="#8172B3", lw=2.4)
    axes[1].axvline(43, ls="--", color="black")
    axes[1].set_ylabel("graph density")
    axes[1].set_xlabel("time step $\\tau$")
    return savefig("02_graph_stability.png")


def plot_drift_and_separability(drift: pd.DataFrame, sep: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))

    axes[0].plot(drift["tau"], drift["mmd"], marker="o", color="#C44E52", label="MMD")
    axes[0].set_ylabel("MMD")
    axes[0].set_xlabel("transition index $\\tau$")
    axes[0].axvline(43, ls="--", color="black", lw=1.5, label="$\\tau=43$")
    ax0b = axes[0].twinx()
    ax0b.plot(drift["tau"], drift["wasserstein_pca"], marker="s", color="#4C72B0", label="Wasserstein PCA")
    ax0b.set_ylabel("Wasserstein on PCA")
    ax0b.grid(False)
    axes[0].set_title("Feature drift is delayed; the largest shift follows the shock")
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax0b.get_legend_handles_labels()
    axes[0].legend(lines + lines2, labels + labels2, loc="upper left")

    sep_agg = (
        sep.groupby(["tau", "Representation"], as_index=False)
        .agg(separable_rate=("separable", "mean"), n_illicit=("n_illicit", "mean"))
        .sort_values(["Representation", "tau"])
    )
    for rep, color in [("Raw", "#4C72B0"), ("Prop_k1_Dir0", "#55A868")]:
        sub = sep_agg[sep_agg["Representation"] == rep]
        label = "raw features" if rep == "Raw" else "1-hop propagated"
        axes[1].plot(sub["tau"], sub["separable_rate"], marker="o", lw=1.7, color=color, label=label)
    axes[1].axvline(43, ls="--", color="black", lw=1.5)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("time step $\\tau$")
    axes[1].set_ylabel("fraction of seeds with separability $p<0.05$")
    axes[1].set_title("At $\\tau=43$, propagated features remain separable")
    axes[1].legend(loc="lower right")
    return savefig("03_drift_separability.png")


def plot_embeddings(pca: pd.DataFrame, tsne: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {0: "#4C72B0", 1: "#C44E52"}
    labels = {0: "licit", 1: "illicit"}
    for label in [0, 1]:
        sub = pca[pca["label"] == label]
        axes[0].scatter(sub["pca1"], sub["pca2"], s=8, alpha=0.38, c=colors[label], label=labels[label])
    axes[0].set_title("PCA: illicit nodes occupy a narrow linear region")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    # t-SNE has thousands of points; deterministic sample for legibility.
    tsne_plot = pd.concat(
        [
            group.sample(min(len(group), 2200), random_state=7)
            for _, group in tsne.groupby("label", sort=False)
        ],
        ignore_index=True,
    )
    for label in [0, 1]:
        sub = tsne_plot[tsne_plot["label"] == label]
        axes[1].scatter(sub["tsne1"], sub["tsne2"], s=8, alpha=0.38, c=colors[label], label=labels[label])
    axes[1].set_title("t-SNE: illicit behavior appears in local pockets")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend()
    return savefig("04_embeddings.png")


def plot_homophily_degree(hom: pd.DataFrame, degree: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    axes[0].plot(hom["tau"], hom["illicit_illicit"], label="illicit-illicit", color="#C44E52", lw=2)
    axes[0].plot(hom["tau"], hom["illicit_licit"], label="illicit-licit", color="#4C72B0", lw=2)
    axes[0].plot(hom["tau"], hom["illicit_unknown"], label="illicit-unknown", color="#8172B3", lw=2)
    axes[0].axvline(43, ls="--", color="black", lw=1.5)
    axes[0].set_xlabel("time step $\\tau$")
    axes[0].set_ylabel("edge count")
    axes[0].set_title("Illicit edges mostly touch unknown intermediaries")
    axes[0].legend()

    clipped = degree.copy()
    clipped["out_degree_clip"] = clipped["out_degree"].clip(upper=12)
    bins = np.arange(-0.5, 12.6, 1)
    for label, color, name in [(0, "#4C72B0", "licit"), (1, "#C44E52", "illicit")]:
        sub = clipped[clipped["label"] == label]
        axes[1].hist(sub["out_degree_clip"], bins=bins, density=True, alpha=0.55, color=color, label=name)
    axes[1].set_xlabel("out-degree, clipped at 12")
    axes[1].set_ylabel("density")
    axes[1].set_title("Illicit out-degree is structurally constrained")
    axes[1].legend()
    return savefig("05_homophily_degree.png")


def plot_intrinsic_dim(intrinsic: pd.DataFrame) -> str:
    df = intrinsic.copy()
    df["PCA_label"] = np.where(df["PCA"], "PCA", "Base")
    grouped = df.groupby(["K", "PCA_label"], as_index=False)["Intrinsic Dimension"].mean()
    pivot = grouped.pivot(index="K", columns="PCA_label", values="Intrinsic Dimension").sort_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for col, color in [("Base", "#4C72B0"), ("PCA", "#C44E52")]:
        if col in pivot:
            ax.plot(pivot.index, pivot[col], marker="o", lw=2.6, label=col, color=color)
    ax.set_xlabel("propagation depth $K$")
    ax.set_ylabel("mean intrinsic dimension")
    ax.set_title("Intrinsic dimensionality exposes oversmoothing pressure")
    ax.legend()
    return savefig("06_intrinsic_dimension.png")


def best_row(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return df.loc[s.idxmax()]


def selected_static_rows(final: pd.DataFrame, phase_d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    picks = {
        "RandomForest": "Baseline: RandomForest (166)",
        "XGBoost": "Baseline: XGBoost WF (epsilon-fallback)",
        "PyG GCN": "Baseline: PyG GCN (2-layer)",
        "Old SGC+MLP": "Sweep 2: + MLP Head",
        "Best old Grid": "Grid: K=3, Dir=T, Topo=None",
    }
    for label, sweep in picks.items():
        sub = final[final["Sweep"].eq(sweep)]
        if sub.empty and label == "Best old Grid":
            sub = final[final["Sweep"].eq(sweep) & final["Variation"].eq("PCA")]
        if sub.empty:
            sub = final[final["Sweep"].astype(str).str.contains(sweep, regex=False)]
        if not sub.empty:
            if label == "Best old Grid":
                row = sub[sub["Variation"].eq("PCA")].iloc[0] if (sub["Variation"].eq("PCA")).any() else sub.iloc[0]
            else:
                row = sub.iloc[0]
            rows.append(
                {
                    "Model": label,
                    "OOT Macro F1": row.get("Static_OOT_Macro_F1_mean"),
                    "OOT Macro PR-AUC": row.get("Static_OOT_Macro_PRAUC_mean"),
                }
            )
    phase_best = best_row(phase_d, "OOT_Macro_PRAUC_mean")
    rows.append(
        {
            "Model": "Final LN+SiLU MLP",
            "OOT Macro F1": phase_best["OOT_Macro_F1_mean"],
            "OOT Macro PR-AUC": phase_best["OOT_Macro_PRAUC_mean"],
        }
    )
    return pd.DataFrame(rows)




def plot_wf_regimes(final: pd.DataFrame) -> str:
    wanted = [
        ("SGC baseline", "Best WF: Sweep 1: SGC (baseline)", "Base"),
        ("SGC + MLP", "Best WF: Sweep 2: + MLP Head", "Base"),
        ("Best SGC WF", "Best WF: Grid: K=2, Dir=F, Topo=None", "Base"),
        ("Best graph + decay", "Ablation: Decay λ=0.25 on 2 T early Base", "Base"),
        ("XGBoost WF", "Baseline: XGBoost WF (epsilon-fallback)", "Base"),
        ("XGBoost + decay", "Ablation: Decay λ=0.25 on XGBoost", "Base"),
    ]
    rows = []
    for label, sweep, variation in wanted:
        sub = final[(final["Sweep"] == sweep) & (final["Variation"] == variation)]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append(
                {
                    "Model": label,
                    "WF Macro PR-AUC": r["WF_Macro_PRAUC_mean"],
                }
            )
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar(x, df["WF Macro PR-AUC"], 0.6, label="WF Macro PR-AUC", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("WF Macro PR-AUC")
    ax.set_title("Walk-forward performance standardized to Macro PR-AUC")
    ax.legend()
    return savefig("08_wf_regimes.png")


def plot_decay(final: pd.DataFrame) -> str:
    rows = []
    for model_label, suffix in [
        ("XGBoost", "XGBoost"),
        ("SGC K=2 Dir=T Topo=early", "2 T early Base"),
        ("SGC K=2 Dir=T Topo=late", "2 T late Base"),
    ]:
        if model_label == "XGBoost":
            base = final[final["Sweep"].eq("Baseline: XGBoost WF (epsilon-fallback)")].iloc[0]
            rows.append({"Model": model_label, "lambda": 0.0, "WF Macro PR-AUC": base["WF_Macro_PRAUC_mean"]})
        else:
            # no-decay walk-forward rows are named "Best WF: Grid: ..."
            topo = "early" if "early" in suffix else "late"
            base_sweep = f"Best WF: Grid: K=2, Dir=T, Topo={topo}"
            base = final[(final["Sweep"].eq(base_sweep)) & (final["Variation"].eq("Base"))]
            if not base.empty:
                rows.append({"Model": model_label, "lambda": 0.0, "WF Macro PR-AUC": base.iloc[0]["WF_Macro_PRAUC_mean"]})
        for lam in [0.05, 0.25, 0.5]:
            sweep = f"Ablation: Decay λ={lam} on {suffix}"
            sub = final[final["Sweep"].eq(sweep)]
            if not sub.empty:
                rows.append({"Model": model_label, "lambda": lam, "WF Macro PR-AUC": sub.iloc[0]["WF_Macro_PRAUC_mean"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for model, color in zip(df["Model"].unique(), ["#4C72B0", "#C44E52", "#55A868"]):
        sub = df[df["Model"] == model].sort_values("lambda")
        ax.plot(sub["lambda"], sub["WF Macro PR-AUC"], marker="o", lw=2.6, label=model, color=color)
    ax.set_xlabel("temporal decay $\\lambda$")
    ax.set_ylabel("WF Macro PR-AUC")
    ax.set_title("Temporal decay under walk-forward Macro PR-AUC")
    ax.legend()
    return savefig("09_temporal_decay.png")


def compute_deep_oot_macro_table(data: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    final = data["final"]
    grid = final[final["Sweep"].astype(str).str.startswith("Grid:")].copy()
    metrics = [
        ("OOT Macro F1", "Static_OOT_Macro_F1_mean", "OOT_Macro_F1"),
        ("OOT Macro PR-AUC", "Static_OOT_Macro_PRAUC_mean", "OOT_Macro_PRAUC"),
    ]
    rows = []
    values = []
    for label, old_col, run_col in metrics:
        grid[old_col] = pd.to_numeric(grid[old_col], errors="coerce")
        old = float(grid.loc[grid[old_col].idxmax(), old_col])
        row = [label, old]
        value_row = {"Metric": label, "Old": old}

        for run in range(4):
            df = data[f"run{run}"].copy()
            for col in [run_col]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if run == 3:
                sub = df[df["Head"].eq("ln_silu_tapered_no_residual")]
            elif run == 2:
                sub = df[df["Head"].eq("ln_silu_small_no_residual")]
            else:
                sub = df[df["Head"].ne("previous_mlp_head")]
            val = float(sub[run_col].max())
            row.extend([val, val - old])
            value_row[f"run{run}"] = val
            value_row[f"run{run}_delta"] = val - old

        for phase in "ABCD":
            df = data[f"phase{phase}"].copy()
            col = run_col + "_mean"
            df[col] = pd.to_numeric(df[col], errors="coerce")
            val = float(df[col].max())
            row.extend([val, val - old])
            value_row[f"sweep{phase}"] = val
            value_row[f"sweep{phase}_delta"] = val - old
        rows.append(row)
        values.append(value_row)

    headers = [
        "Metric",
        "Old",
        "run0",
        "Δ0",
        "run1",
        "Δ1",
        "run2",
        "Δ2",
        "run3",
        "Δ3",
        "sweepA",
        "ΔA",
        "sweepB",
        "ΔB",
        "sweepC",
        "ΔC",
        "sweepD",
        "ΔD",
    ]

    table_rows = []
    for row in rows:
        out = [row[0], f"{row[1]:.6f}"]
        for i in range(2, len(row), 2):
            out.append(f"{row[i]:.6f}")
            out.append(f"{row[i + 1]:+.6f}")
        table_rows.append(out)

    md = markdown_table(headers, table_rows)
    return md, pd.DataFrame(values)


def plot_deep_oot_macro_heatmap(values: pd.DataFrame) -> str:
    values = values[values["Metric"].eq("OOT Macro PR-AUC")].reset_index(drop=True)
    metrics = values["Metric"].tolist()
    cols = ["run0", "run1", "run2", "run3", "sweepA", "sweepB", "sweepC", "sweepD"]
    deltas = np.array([[row[f"{col}_delta"] for col in cols] for _, row in values.iterrows()])
    vmax = max(abs(np.nanmin(deltas)), abs(np.nanmax(deltas)))
    fig, ax = plt.subplots(figsize=(11, 2.6))
    im = ax.imshow(deltas, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            ax.text(j, i, f"{deltas[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("OOT Macro PR-AUC deltas versus the old Grid MLP benchmark")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="new - old")
    return savefig("10_deep_mlp_oot_macro_deltas.png")




def build_summary_tables(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    snapshot = data["snapshot"]
    rows = []
    for tau in [42, 43, 44, 45, 46, 49]:
        r = snapshot[snapshot["Tau"] == tau].iloc[0]
        rows.append([tau, r["N_nodes"], r["N_illicit"], r["Illicit_Rate"], r["Mean_Degree"], r["Regime"]])
    shock_table = markdown_table(
        ["$\\tau$", "nodes", "illicit", "illicit rate", "mean degree", "regime"], rows
    )

    drift = data["drift"]
    drift_rows = []
    for tau in [42, 43, 44, 45, 46]:
        r = drift[drift["tau"] == tau].iloc[0]
        drift_rows.append([tau, r["mmd"], r["wasserstein_pca"]])
    drift_table = markdown_table(["$\\tau$", "MMD", "Wasserstein PCA"], drift_rows, digits=4)

    sep = data["separability"]
    sep_rows = []
    for tau in [42, 43, 44, 45]:
        for rep in ["Raw", "Prop_k1_Dir0"]:
            sub = sep[(sep["tau"] == tau) & (sep["Representation"] == rep)]
            if not sub.empty:
                sep_rows.append(
                    [
                        tau,
                        "Raw" if rep == "Raw" else "$\\tilde A X$",
                        sub["n_illicit"].mean(),
                        sub["separable"].mean(),
                        sub["perm_p"].median(),
                    ]
                )
    sep_table = markdown_table(["$\\tau$", "representation", "n illicit", "separable seed fraction", "median perm. p"], sep_rows, digits=4)

    final = data["final"]
    static_rows = []
    for model, sweep, variation in [
        ("RandomForest", "Baseline: RandomForest (166)", "Base"),
        ("XGBoost", "Baseline: XGBoost WF (epsilon-fallback)", "Base"),
        ("PyG GCN", "Baseline: PyG GCN (2-layer)", "Base"),
        ("Old SGC+MLP", "Sweep 2: + MLP Head", "Base"),
        ("Best old Graph Grid", "Grid: K=3, Dir=T, Topo=None", "PCA"),
    ]:
        sub = final[(final["Sweep"] == sweep) & (final["Variation"] == variation)]
        if not sub.empty:
            r = sub.iloc[0]
            static_rows.append(
                [
                    model,
                    r["Static_OOT_Macro_F1_mean"],
                    r["Static_OOT_Macro_PRAUC_mean"],
                ]
            )
    deep_candidates = []
    for phase in "ABCD":
        df = data[f"phase{phase}"].copy()
        df["Phase"] = phase
        deep_candidates.append(df)
    deep_sweeps = pd.concat(deep_candidates, ignore_index=True)
    r = best_row(deep_sweeps, "OOT_Macro_PRAUC_mean")
    static_rows.append(
        [
            f"Best LN+SiLU MLP (Phase {r['Phase']})",
            r["OOT_Macro_F1_mean"],
            r["OOT_Macro_PRAUC_mean"],
        ]
    )
    static_table = markdown_table(
        ["model", "OOT Macro F1", "OOT Macro PR-AUC"],
        static_rows,
    )

    wf_rows = []
    for model, sweep, variation in [
        ("SGC baseline", "Best WF: Sweep 1: SGC (baseline)", "Base"),
        ("SGC + MLP", "Best WF: Sweep 2: + MLP Head", "Base"),
        ("Best SGC WF", "Best WF: Grid: K=2, Dir=F, Topo=None", "Base"),
        ("Best graph + decay", "Ablation: Decay λ=0.25 on 2 T early Base", "Base"),
        ("XGBoost WF", "Baseline: XGBoost WF (epsilon-fallback)", "Base"),
        ("XGBoost + decay", "Ablation: Decay λ=0.25 on XGBoost", "Base"),
    ]:
        sub = final[(final["Sweep"] == sweep) & (final["Variation"] == variation)]
        if not sub.empty:
            r = sub.iloc[0]
            wf_rows.append(
                [
                    model,
                    r["WF_Macro_F1_mean"],
                    r["WF_Macro_PRAUC_mean"],
                ]
            )
    wf_table = markdown_table(
        ["model", "WF Macro F1", "WF Macro PR-AUC"], wf_rows
    )

    return {
        "shock_table": shock_table,
        "drift_table": drift_table,
        "sep_table": sep_table,
        "static_table": static_table,
        "wf_table": wf_table,
    }


def md_cell(text: str, slide_type: str = "slide"):
    cell = nbf.v4.new_markdown_cell(text.strip() + "\n")
    cell["metadata"]["slideshow"] = {"slide_type": slide_type}
    return cell


def code_cell(code: str, slide_type: str = "fragment"):
    cell = nbf.v4.new_code_cell(code.strip() + "\n")
    cell["metadata"]["slideshow"] = {"slide_type": slide_type}
    return cell


def build_notebook(data: dict[str, pd.DataFrame], assets: dict[str, str], tables: dict[str, str], deep_table: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (xai_project)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "celltoolbar": "Slideshow",
    }

    def read_narrative(filename: str) -> str:
        path = SOURCE / "reporting" / "results" / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    cells = []
    
    cells.append(
        md_cell(
            """
# Detecting Illicit Bitcoin Transactions in a Temporal Graph

## Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection

**Dataset:** Elliptic Bitcoin transaction graph  
**Task:** detect illicit transactions under temporal non-stationarity  

Generated from:
- `results/` experiment outputs
- `source/` implementation
- `source/reporting/results/` analysis summaries
"""
        )
    )

    narrative = read_narrative("executive_summary.md")
    if narrative:
        cells.append(md_cell(narrative))

    cells.append(
        md_cell(
            """
## Evaluation metrics

Because illicit transactions are rare, accuracy is not informative. We track metrics for the positive class, where positive means **illicit**:

$$
\\mathrm{Precision}=\\frac{TP}{TP+FP},
\\qquad
\\mathrm{Recall}=\\frac{TP}{TP+FN},
\\qquad
F_1=\\frac{2\\,\\mathrm{Precision}\\,\\mathrm{Recall}}{\\mathrm{Precision}+\\mathrm{Recall}}.
$$

We also use **PR-AUC**, the area under the precision--recall curve. PR-AUC is threshold-free, so it measures ranking quality even when the fixed classification threshold is imperfect.

The presentation reports **Macro** metrics in the main figures and tables: compute the metric separately at each timestep, then average across timesteps.

This is the stricter convention for our data because a single large timestep cannot dominate the score. It is also the fairest way to discuss the post-shutdown snapshots, where the dataset becomes temporally uneven.
"""
        )
    )

    cells.append(
        md_cell(
            """
## Experimental protocol and leakage guards

The project uses two complementary evaluation protocols.

### Static out-of-time protocol

$$
\\text{train}: \\tau=1\\ldots 26,
\\qquad
\\text{development}: \\tau=27\\ldots 34,
\\qquad
\\text{test/OOT}: \\tau=35\\ldots 49.
$$

The development split exists in the codebase, but the presentation does **not** use it as the main performance claim because it stops before the shutdown/recovery period. Main static comparisons use OOT Macro metrics on $\\tau=35\\ldots49$.

### Walk-forward protocol

At each test step $\\tau$:

$$
\\text{train on }[1,\\tau-2],
\\qquad
\\text{calibrate threshold on }\\tau-1,
\\qquad
\\text{test on }\\tau.
$$

This matters mathematically: the threshold is calibrated on a held-out previous snapshot, not on the test snapshot itself.
"""
        )
    )

    if "eda_panel_b_volume" in assets:
        cells.append(md_cell(f"![Transaction Volume per Snapshot]({assets['eda_panel_b_volume']})"))

    narrative = read_narrative("snapshot_topology_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
        
    if "panel1_ground_truth" in assets:
        cells.append(md_cell(f"![Ground truth timeline]({assets['panel1_ground_truth']})"))

    cells.append(md_cell(f"### The temporal regime split\n\n{tables['shock_table']}"))
    cells.append(md_cell(f"![Graph stability]({assets['graph_stability']})"))

    narrative = read_narrative("eda_degree_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    
    narrative = read_narrative("eda_embeddings_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    cells.append(md_cell(f"![Embeddings]({assets['embeddings']})"))

    narrative = read_narrative("eda_pagerank_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    
    if "eda_panel_c_hairball" in assets:
        cells.append(md_cell(f"![PCA+TSNE+PageRank]({assets['eda_panel_c_hairball']})"))

    narrative = read_narrative("eda_homophily_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    cells.append(md_cell(f"![Homophily and degree]({assets['homophily_degree']})"))

    narrative = read_narrative("diagnostic_falsification_report.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    cells.append(md_cell(f"### Feature drift around the shock\n\n{tables['drift_table']}"))
    cells.append(md_cell(f"### Permutation separability tests\n\n{tables['sep_table']}"))
    cells.append(md_cell(f"![Drift and separability]({assets['drift_sep']})"))

    narrative = read_narrative("baseline_performance_report.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    if "cost_vs_perf" in assets:
        cells.append(md_cell(f"![Cost vs Performance]({assets['cost_vs_perf']})"))
    cells.append(md_cell(f"### Static OOT Macro result table\n\n{tables['static_table']}"))

    narrative = read_narrative("sgc_grid_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    if "sgc_oversmoothing" in assets:
        cells.append(md_cell(f"![PCA as Oversmoothing Regularizer]({assets['sgc_oversmoothing']})"))
        cells.append(md_cell("> **NOTE**: PCA here = input-compression regularizer (reduces oversmoothing at K=3). This is distinct from the drift-diagnostic PCA in Section 2."))
    
    narrative = read_narrative("deep_res_mlp_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
        cells.append(md_cell(f"![MLP OOT Macro PR-AUC deltas]({assets['deep_heatmap']})"))
        cells.append(md_cell(f"### OOT Macro deltas versus the old Grid MLP benchmark\n\n{deep_table}"))

    narrative = read_narrative("wf_temporal_analysis.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))
    cells.append(md_cell(f"![WF regimes]({assets['wf_regimes']})"))
    cells.append(md_cell(f"### Walk-forward result table\n\n{tables['wf_table']}"))
    cells.append(md_cell(f"![Temporal decay]({assets['decay']})"))

    cells.append(
        md_cell(
            """
---
## What won?

| Question | Answer |
|---|---|
| Best overall model | XGBoost / XGBoost + temporal decay |
| Best old SGC grid configuration | $K=3$, directional propagation, PCA, no explicit topology |
| Best final graph head | $K=3$, directional propagation, PCA, late topology, LayerNorm + SiLU + `(64, 64)`, **no residual** |
| Main diagnostic finding | $\\tau=43$ is prior shift, not representational collapse |
| Main graph-learning failure mode | topological overfitting to pre-shock micro-motifs |
| Most useful temporal fix | walk-forward training with exponential decay |

The most important negative result is also important: the graph models improved substantially, but the best tabular models still dominate the global benchmark.
"""
        )
    )

    cells.append(
        md_cell(
            """
---
## Limitations

1. **Unknown labels dominate the graph.** Unknown nodes are excluded from loss and metrics, but many illicit-adjacent edges pass through unknown intermediaries.
2. **Static MLP-head sweeps are not full deployment simulations.** They are reported with OOT Macro metrics, but they were not fully rerun under the complete walk-forward regime.
3. **PR-AUC and F1 answer different questions.** Some MLP variants improve ranking quality while worsening fixed-threshold F1.
4. **Discrete snapshots simplify blockchain time.** A native temporal graph model may represent transaction time more faithfully than 49 snapshot graphs.
5. **Causal attribution is limited.** The $\\tau=43$ interpretation is consistent with AlphaBay-era timing and label dynamics, but the dataset is anonymized.

These limitations do not invalidate the results; they clarify exactly what the experiments prove.
"""
        )
    )

    narrative = read_narrative("conclusion.md")
    if narrative:
        cells.append(md_cell(f"---\n{narrative}"))

    cells.append(
        md_cell(
            """
---
## Implementation map

Important source files checked for this presentation:

| File | Role |
|---|---|
| `source/data/load_dataset.py` | loads Elliptic data and validates temporal edge integrity |
| `source/data/build_graph.py` | builds per-timestep graphs, scales features, injects topology, applies PCA |
| `source/models/layers.py` | SGC propagation, multiscale concatenation, directional channels |
| `source/models/classifier.py` | MLP head, LayerNorm, SiLU/ReLU activation checks, residual projection |
| `source/evaluation/validation.py` | static and walk-forward evaluation, threshold calibration |
| `source/evaluation/ablation_validation.py` | temporal decay and additional walk-forward ablations |
| `source/sweep.py` | experiment orchestration and result schema |
| `source/reporting/results/*.md` | written analyses used to shape the presentation narrative |
"""
        )
    )

    cells.append(
        code_cell(
            """
# Reproducibility entry point
# The notebook's plots were generated by:
#   python source/reporting/build_presentation_notebook.py
#
# Main result folders:
#   results/final_aggregated_results.csv
#   results/final_aggregated_timesteps.csv
#   results/deep_res_mlp_results/sweep_phaseA
#   results/deep_res_mlp_results/sweep_phaseB
#   results/deep_res_mlp_results/sweep_phaseC
#   results/deep_res_mlp_results/sweep_phaseD
"""
        )
    )

    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK)


def main() -> None:
    ensure_dirs()
    data = load_data()
    copied = copy_existing_figures()

    assets = {
        "prior_shift": plot_prior_shift(data["snapshot"]),
        "graph_stability": plot_graph_stability(data["snapshot"]),
        "drift_sep": plot_drift_and_separability(data["drift"], data["separability"]),
        "embeddings": plot_embeddings(data["pca"], data["tsne"]),
        "homophily_degree": plot_homophily_degree(data["homophily"], data["degree"]),
        "intrinsic": plot_intrinsic_dim(data["intrinsic"]),
        "wf_regimes": plot_wf_regimes(data["final"]),
        "decay": plot_decay(data["final"]),
        "sgc_oversmoothing": plot_sgc_oversmoothing(data["final"]),
        "cost_vs_perf": plot_cost_vs_performance(data["sweep"]),
    }
    assets.update(copied)
    deep_table, deep_values = compute_deep_oot_macro_table(data)
    assets["deep_heatmap"] = plot_deep_oot_macro_heatmap(deep_values)
    tables = build_summary_tables(data)

    build_notebook(data, assets, tables, deep_table)

    manifest = {
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "assets_dir": str(ASSETS.relative_to(ROOT)),
        "asset_count": len(list(ASSETS.glob("*.png"))),
        "sources": [
            "results/*.csv",
            "results/deep_res_mlp_results/**",
            "source/**",
            "source/reporting/results/*.md",
        ],
    }
    (OUT_DIR / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
