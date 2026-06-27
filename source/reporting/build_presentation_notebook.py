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


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "snapshot": read_csv("snapshot_topology.csv"),
        "drift": read_csv("eda_drift.csv"),
        "separability": read_csv("label_separability.csv"),
        "homophily": read_csv("eda_homophily.csv"),
        "degree": read_csv("eda_degree.csv"),
        "degree_stats": read_csv("eda_degree_stats.csv"),
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
                    "OOT Pooled F1": row.get("Static_OOT_Pooled_F1_mean"),
                    "OOT Pooled PR-AUC": row.get("Static_OOT_Pooled_PRAUC_mean"),
                    "Val Macro PR-AUC": row.get("Static_Val_Macro_PRAUC_mean"),
                }
            )
    phase_best = best_row(phase_d, "Val_Macro_PRAUC_mean")
    rows.append(
        {
            "Model": "Final LN+SiLU MLP",
            "OOT Pooled F1": phase_best["OOT_Pooled_F1_mean"],
            "OOT Pooled PR-AUC": phase_best["OOT_Pooled_PRAUC_mean"],
            "Val Macro PR-AUC": phase_best["Val_Macro_PRAUC_mean"],
        }
    )
    return pd.DataFrame(rows)


def plot_static_results(static_df: pd.DataFrame) -> str:
    df = static_df.copy()
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - width / 2, df["OOT Pooled F1"], width, label="OOT pooled F1", color="#4C72B0")
    ax.bar(x + width / 2, df["OOT Pooled PR-AUC"], width, label="OOT pooled PR-AUC", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 0.88)
    ax.set_title("Static out-of-time results: tabular baselines remain strongest")
    ax.legend()
    return savefig("07_static_results.png")


def plot_wf_regimes(final: pd.DataFrame) -> str:
    wanted = [
        ("SGC baseline", "Best WF: Sweep 1: SGC (baseline)", "Base"),
        ("SGC + MLP", "Best WF: Sweep 2: + MLP Head", "Base"),
        ("Best SGC WF", "Best WF: Grid: K=2, Dir=F, Topo=None", "Base"),
        ("Best graph recovery", "Best WF: Grid: K=3, Dir=T, Topo=early", "PCA"),
        ("XGBoost WF", "Baseline: XGBoost WF (epsilon-fallback)", "Base"),
        ("XGBoost + decay", "Ablation: Decay λ=0.5 on XGBoost", "Base"),
    ]
    rows = []
    for label, sweep, variation in wanted:
        sub = final[(final["Sweep"] == sweep) & (final["Variation"] == variation)]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append(
                {
                    "Model": label,
                    "Pre-shock": r["WF_Pre43_Pooled_F1_mean"],
                    "Shock": r["WF_Shock_F1_mean"],
                    "Recovery": r["WF_Recovery_Pooled_F1_mean"],
                }
            )
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar(x - width, df["Pre-shock"], width, label="Pre-shock", color="#4C72B0")
    ax.bar(x, df["Shock"], width, label="Shock", color="#C44E52")
    ax.bar(x + width, df["Recovery"], width, label="Recovery", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("pooled illicit F1")
    ax.set_title("Walk-forward regime breakdown reveals the graph recovery trap")
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
            rows.append({"Model": model_label, "lambda": 0.0, "Recovery F1": base["WF_Recovery_Pooled_F1_mean"]})
        else:
            # no-decay walk-forward rows are named "Best WF: Grid: ..."
            topo = "early" if "early" in suffix else "late"
            base_sweep = f"Best WF: Grid: K=2, Dir=T, Topo={topo}"
            base = final[(final["Sweep"].eq(base_sweep)) & (final["Variation"].eq("Base"))]
            if not base.empty:
                rows.append({"Model": model_label, "lambda": 0.0, "Recovery F1": base.iloc[0]["WF_Recovery_Pooled_F1_mean"]})
        for lam in [0.05, 0.25, 0.5]:
            sweep = f"Ablation: Decay λ={lam} on {suffix}"
            sub = final[final["Sweep"].eq(sweep)]
            if not sub.empty:
                rows.append({"Model": model_label, "lambda": lam, "Recovery F1": sub.iloc[0]["WF_Recovery_Pooled_F1_mean"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for model, color in zip(df["Model"].unique(), ["#4C72B0", "#C44E52", "#55A868"]):
        sub = df[df["Model"] == model].sort_values("lambda")
        ax.plot(sub["lambda"], sub["Recovery F1"], marker="o", lw=2.6, label=model, color=color)
    ax.set_xlabel("temporal decay $\\lambda$")
    ax.set_ylabel("recovery pooled F1")
    ax.set_title("Temporal decay improves recovery by forgetting stale topology")
    ax.legend()
    return savefig("09_temporal_decay.png")


def compute_deep_validation_table(data: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    final = data["final"]
    grid = final[final["Sweep"].astype(str).str.startswith("Grid:")].copy()
    metrics = [
        ("Val Macro F1", "Static_Val_Macro_F1_mean", "Val_Macro_F1"),
        ("Val Pooled F1", "Static_Val_Pooled_F1_mean", "Val_Pooled_F1"),
        ("Val Macro PR-AUC", "Static_Val_Macro_PRAUC_mean", "Val_Macro_PRAUC"),
        ("Val Pooled PR-AUC", "Static_Val_Pooled_PRAUC_mean", "Val_Pooled_PRAUC"),
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


def plot_deep_validation_heatmap(values: pd.DataFrame) -> str:
    metrics = values["Metric"].tolist()
    cols = ["run0", "run1", "run2", "run3", "sweepA", "sweepB", "sweepC", "sweepD"]
    deltas = np.array([[row[f"{col}_delta"] for col in cols] for _, row in values.iterrows()])
    vmax = max(abs(np.nanmin(deltas)), abs(np.nanmax(deltas)))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(deltas, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            ax.text(j, i, f"{deltas[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Validation deltas versus the old Grid MLP benchmark")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="new - old")
    return savefig("10_deep_mlp_validation_deltas.png")


def plot_phase_sweeps(data: dict[str, pd.DataFrame]) -> str:
    rows = []
    for phase in "ABCD":
        df = data[f"phase{phase}"].copy()
        for col in ["Val_Macro_PRAUC_mean", "Val_Pooled_PRAUC_mean", "OOT_Pooled_PRAUC_mean"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        r = df.loc[df["Val_Macro_PRAUC_mean"].idxmax()]
        rows.append(
            {
                "Phase": phase,
                "Val Macro PR-AUC": r["Val_Macro_PRAUC_mean"],
                "Val Pooled PR-AUC": r["Val_Pooled_PRAUC_mean"],
                "OOT Pooled PR-AUC": r["OOT_Pooled_PRAUC_mean"],
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, color in [
        ("Val Macro PR-AUC", "#C44E52"),
        ("Val Pooled PR-AUC", "#4C72B0"),
        ("OOT Pooled PR-AUC", "#55A868"),
    ]:
        ax.plot(df["Phase"], df[col], marker="o", lw=2.6, label=col, color=color)
    ax.set_ylim(0.25, 0.98)
    ax.set_xlabel("sweep phase")
    ax.set_title("Final MLP-head sweep: gains came from small LN+SiLU, not residuals")
    ax.legend(loc="lower right")
    return savefig("11_phase_sweeps.png")


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
                    r["Static_Val_Macro_PRAUC_mean"],
                    r["Static_OOT_Pooled_F1_mean"],
                    r["Static_OOT_Pooled_PRAUC_mean"],
                    r["Static_OOT_Macro_F1_mean"],
                ]
            )
    phase_d = data["phaseD"]
    r = best_row(phase_d, "Val_Macro_PRAUC_mean")
    static_rows.append(
        [
            "Final LN+SiLU MLP",
            r["Val_Macro_PRAUC_mean"],
            r["OOT_Pooled_F1_mean"],
            r["OOT_Pooled_PRAUC_mean"],
            r["OOT_Macro_F1_mean"],
        ]
    )
    static_table = markdown_table(
        ["model", "Val Macro PR-AUC", "OOT pooled F1", "OOT pooled PR-AUC", "OOT macro F1"],
        static_rows,
    )

    wf_rows = []
    for model, sweep, variation in [
        ("SGC baseline", "Best WF: Sweep 1: SGC (baseline)", "Base"),
        ("SGC + MLP", "Best WF: Sweep 2: + MLP Head", "Base"),
        ("Best SGC WF", "Best WF: Grid: K=2, Dir=F, Topo=None", "Base"),
        ("Best graph recovery", "Best WF: Grid: K=3, Dir=T, Topo=early", "PCA"),
        ("XGBoost WF", "Baseline: XGBoost WF (epsilon-fallback)", "Base"),
        ("XGBoost + decay", "Ablation: Decay λ=0.5 on XGBoost", "Base"),
    ]:
        sub = final[(final["Sweep"] == sweep) & (final["Variation"] == variation)]
        if not sub.empty:
            r = sub.iloc[0]
            wf_rows.append(
                [
                    model,
                    r["WF_Pooled_F1_mean"],
                    r["WF_Macro_F1_mean"],
                    r["WF_Pre43_Pooled_F1_mean"],
                    r["WF_Shock_F1_mean"],
                    r["WF_Recovery_Pooled_F1_mean"],
                ]
            )
    wf_table = markdown_table(
        ["model", "WF pooled F1", "WF macro F1", "pre-shock F1", "shock F1", "recovery F1"], wf_rows
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

    cells = []
    cells.append(
        md_cell(
            """
# Detecting Illicit Bitcoin Transactions in a Temporal Graph

## Geometric Learning, Time-Variant Data Analysis, and Anomaly Detection

**Dataset:** Elliptic Bitcoin transaction graph  
**Task:** detect illicit transactions under temporal non-stationarity  
**Main thesis:** the catastrophic post-shock failure is mostly a *prior-shift / topology-adaptation problem*, not a collapse of the node representation.

Generated from:

- `results/` experiment outputs
- `source/` implementation
- `source/reporting/results/` analysis summaries
"""
        )
    )

    cells.append(
        md_cell(
            """
## Executive summary

1. The Elliptic graph is a sequence of directed transaction snapshots $G_\\tau=(V_\\tau,E_\\tau)$.
2. The shock at $\\tau=43$ is not primarily a geometric collapse. It is a **class-prior collapse**: illicit labels drop from 239 at $\\tau=42$ to 24 at $\\tau=43$.
3. SGC propagation and PCA uncover useful graph structure, but graph models overfit to pre-shock micro-motifs.
4. XGBoost is the strongest overall benchmark because node-level tabular structure survives the regime change better than graph motifs.
5. The final MLP-head experiment improved the old graph-MLP validation PR-AUC, but the residual connection itself did not help. The useful head is **LayerNorm + SiLU + small non-residual MLP**.
"""
        )
    )

    cells.append(
        md_cell(
            """
## Data model and notation

At each time step $\\tau \\in \\{1,\\dots,49\\}$ we have a directed graph

$$
G_\\tau=(V_\\tau,E_\\tau,X_\\tau,y_\\tau),
$$

where each node is a Bitcoin transaction, each edge is a flow of funds, and

$$
y_i \\in \\{0,1,-1\\}
$$

means licit, illicit, or unknown. Unknown labels are excluded from the loss and metrics.

The supervised task is imbalanced: the positive class is illicit, so **PR-AUC** is more informative than accuracy.
"""
        )
    )

    cells.append(
        md_cell(
            f"""
## The temporal regime split

{tables['shock_table']}

The key observation is the discontinuity in illicit prevalence at $\\tau=43$. The global graph is still present; the minority class almost disappears.
"""
        )
    )

    cells.append(md_cell(f"![Prior shift]({assets['prior_shift']})"))
    cells.append(md_cell(f"![Graph stability]({assets['graph_stability']})"))
    if all(k in assets for k in ["panel1_ground_truth", "eda_panel_a_imbalance", "eda_panel_b_volume", "eda_panel_c_hairball"]):
        cells.append(
            md_cell(
                f"""
## Visual diagnostics from the EDA pipeline

These figures were generated earlier in the project and copied from `results/figures/`.

![Ground truth timeline]({assets['panel1_ground_truth']})

![Class imbalance panel]({assets['eda_panel_a_imbalance']})

![Volume panel]({assets['eda_panel_b_volume']})

![Graph hairball panel]({assets['eda_panel_c_hairball']})
"""
            )
        )

    cells.append(
        md_cell(
            """
## Exploratory graph geometry

The raw feature geometry and local graph structure are informative:

- illicit nodes are more structurally constrained downstream;
- illicit interactions often pass through unknown intermediaries;
- licit transactions include high-degree service-like hubs;
- graph homophily is weak for illicit nodes, which makes naive neighbor smoothing risky.
"""
        )
    )

    cells.append(md_cell(f"![Embeddings]({assets['embeddings']})"))
    cells.append(md_cell(f"![Homophily and degree]({assets['homophily_degree']})"))

    cells.append(
        md_cell(
            """
## SGC propagation: the graph signal used by the model

The implementation in `source/models/layers.py` uses a symmetrically normalized adjacency

$$
\\tilde A = D^{-1/2}\\,(\\max(A,A^T)+I)\\,D^{-1/2}.
$$

For multiscale SGC, the representation is

$$
\\Phi_K(X)=\\left[X\\;\\middle|\\;\\tilde A X\\;\\middle|\\;\\tilde A^2X\\;\\middle|\\;\\cdots\\;\\middle|\\;\\tilde A^KX\\right].
$$

The directional version augments the symmetric channel with outgoing and incoming row-normalized channels:

$$
\\Phi_K^{dir}(X)=
\\left[
X,
\\tilde A_{sym}X,\\tilde A_{out}X,\\tilde A_{in}X,
\\dots,
\\tilde A_{sym}^KX,\\tilde A_{out}^KX,\\tilde A_{in}^KX
\\right].
$$

This is not a trainable GCN layer; propagation is deterministic, and the learning happens in the classifier head.
"""
        )
    )

    cells.append(
        md_cell(
            """
## PCA and oversmoothing

Deep propagation increases the amount of neighborhood information mixed into each node. For large $K$, this can cause **oversmoothing**: node representations become too similar.

The project tested intrinsic dimensionality as a diagnostic. A drop in intrinsic dimension at larger $K$ is evidence that representations are collapsing toward a lower-dimensional, less discriminative manifold.
"""
        )
    )
    cells.append(md_cell(f"![Intrinsic dimension]({assets['intrinsic']})"))

    cells.append(
        md_cell(
            f"""
## Falsifying representational collapse at $\\tau=43$

The strongest misconception is: "the graph embedding collapses at the shock." The diagnostic files contradict that.

**Feature drift around the shock:**

{tables['drift_table']}

**Permutation separability tests:**

{tables['sep_table']}

The propagated representation at $\\tau=43$ remains separable. The model failure is therefore more consistent with **label deprivation and threshold/head-level imbalance** than with geometric collapse.
"""
        )
    )
    cells.append(md_cell(f"![Drift and separability]({assets['drift_sep']})"))

    cells.append(
        md_cell(
            """
## Static evaluation: validation and out-of-time tests

The static protocol trains on early timesteps, validates on an intermediate block, and reports out-of-time performance on future timesteps.

This is where graph propagation helps compared with a plain SGC baseline, but tabular tree models remain very strong.
"""
        )
    )
    cells.append(md_cell(f"![Static results]({assets['static_results']})"))
    cells.append(md_cell(f"### Static result table\n\n{tables['static_table']}"))

    cells.append(
        md_cell(
            """
## Walk-forward validation

The implementation in `source/evaluation/validation.py` uses a leakage-guarded walk-forward protocol:

$$
\\text{train on } [1,\\tau-2],\\quad
\\text{calibrate threshold on } \\tau-1,\\quad
\\text{test on } \\tau.
$$

This separates training from threshold selection. When the calibration step has too few positives, the code uses an $\\epsilon$-fallback threshold rule.
"""
        )
    )
    cells.append(md_cell(f"![WF regimes]({assets['wf_regimes']})"))
    cells.append(md_cell(f"### Walk-forward result table\n\n{tables['wf_table']}"))

    cells.append(
        md_cell(
            """
## The graph recovery trap

Graph models can perform well before the shock because they learn the local motifs of the pre-shock illicit economy.

After $\\tau=43$, illicit actors re-enter through different local structures. The graph features learned before the shock become stale. This is **topological overfitting**:

$$
\\text{good pre-shock motif memory} \\;\\not\\Rightarrow\\; \\text{good post-shock generalization}.
$$

The evidence is the recovery gap: XGBoost recovers substantially better than SGC variants, even though graph models can be competitive pre-shock.
"""
        )
    )

    cells.append(
        md_cell(
            """
## Temporal decay

The temporal-decay ablation in `source/evaluation/ablation_validation.py` weights old training examples less:

$$
w_i \\propto \\exp\\{-\\lambda(\\tau-t_i)\\}\\,c(y_i),
$$

where $c(y_i)$ is the class-imbalance multiplier. The mathematical idea is simple: if the graph regime has changed, old topology should not dominate the loss.
"""
        )
    )
    cells.append(md_cell(f"![Temporal decay]({assets['decay']})"))

    cells.append(
        md_cell(
            """
## The final MLP-head experiment

The late experiment tried to improve the graph head on top of multiscale SGC.

The proposed residual idea was tested, but the result was clear:

- wide residual heads overfit and underperform;
- small residual heads still underperform;
- the useful change is **LayerNorm + SiLU + a small non-residual MLP**.

The winning graph-head recipe is therefore:

```python
use_mlp_head = True
use_layernorm = True
activation = "silu"
use_residual = False
mlp_hidden = (64, 64)
mlp_dropout = 0.4
sgc_k = 3
use_directional_prop = True
topo_injection_mode = "late"
use_pca = True
pca_variance = 0.98
sgc_lr = 0.01
sgc_weight_decay = 0.0005
```
"""
        )
    )
    cells.append(md_cell(f"![MLP validation deltas]({assets['deep_heatmap']})"))
    cells.append(md_cell(f"### Validation deltas versus the old Grid MLP benchmark\n\n{deep_table}"))
    cells.append(md_cell(f"![Phase sweeps]({assets['phase_sweeps']})"))

    cells.append(
        md_cell(
            """
## What the final MLP head did and did not prove

**It did prove:**

- LayerNorm + SiLU improves graph-head ranking metrics over the old graph MLP baseline.
- Small heads generalize better than wide heads.
- Residual connections are not automatically beneficial in this setting.

**It did not prove:**

- that the graph model beats the best tabular baseline;
- that F1 improves under the fixed $0.5$ threshold;
- that architecture alone solves the $\\tau=43$ label-deprivation problem.

The final claim should be precise: **we improved the graph MLP ranking performance, but the global benchmark is still dominated by tree-based tabular models.**
"""
        )
    )

    cells.append(
        md_cell(
            """
## Final conclusions

1. The Elliptic Bitcoin task is dominated by temporal non-stationarity and class imbalance.
2. $\\tau=43$ is mainly a prior-shift event: the illicit class nearly disappears.
3. $\\tau\\ge 44$ creates the harder recovery problem: illicit actors return with different micro-structure.
4. SGC and multiscale graph features help, but deep graph propagation risks topological overfitting.
5. PCA regularizes deep propagation and partially rescues $K=3$.
6. Temporal decay is the most principled fix for stale topology.
7. The best final graph head is small, normalized, and non-residual.
8. XGBoost remains the strongest overall model, which is an important negative result for the graph-learning hypothesis.
"""
        )
    )

    cells.append(
        md_cell(
            """
## Implementation map

Important source files checked for this presentation:

| File | Role |
|---|---|
| `source/data/load_dataset.py` | loads Elliptic data and validates temporal edge integrity |
| `source/data/build_graph.py` | builds per-timestep graphs, scales features, injects topology, applies PCA |
| `source/models/layers.py` | SGC propagation, multiscale concatenation, directional channels |
| `source/models/classifier.py` | MLP head, LayerNorm, SiLU/ReLU validation, residual projection |
| `source/evaluation/validation.py` | static and walk-forward validation, threshold calibration |
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
        "static_results": plot_static_results(selected_static_rows(data["final"], data["phaseD"])),
        "wf_regimes": plot_wf_regimes(data["final"]),
        "decay": plot_decay(data["final"]),
        "phase_sweeps": plot_phase_sweeps(data),
    }
    assets.update(copied)
    deep_table, deep_values = compute_deep_validation_table(data)
    assets["deep_heatmap"] = plot_deep_validation_heatmap(deep_values)
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
