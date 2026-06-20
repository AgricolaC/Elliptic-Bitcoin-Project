"""Assemble presentation.ipynb (Structural Architecture Findings)."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ---------------------------------------------------------------- Setup
code(
    "import os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from IPython.display import display, Markdown\n"
    "import tabulate\n\n"
    "RESULTS = 'results'\n"
    "FIGDIR = os.path.join(RESULTS, 'figures')\n"
    "os.makedirs(FIGDIR, exist_ok=True)\n\n"
    "plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 11,\n"
    "                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})\n\n"
    "topo = pd.read_csv(os.path.join(RESULTS, 'snapshot_topology.csv'))\n"
    "sweep = pd.read_csv(os.path.join(RESULTS, 'final_aggregated_results.csv'))\n"
    "steps = pd.read_csv(os.path.join(RESULTS, 'final_aggregated_timesteps.csv'))\n\n"
    "SHOCK = 43\n\n"
    "def get_scalar(sweep_name, col):\n"
    "    col_name = f'{col}_mean'\n"
    "    if col_name not in sweep.columns: col_name = col\n"
    "    hits = sweep.loc[(sweep['Sweep'] == sweep_name) & (sweep['Variation'] == 'Base')]\n"
    "    if len(hits) == 0: hits = sweep.loc[sweep['Sweep'] == sweep_name]\n"
    "    if len(hits) == 0: return 0.0\n"
    "    return float(hits[col_name].iloc[0])\n"
)

# ---------------------------------------------------------------- Slide 1: Title
md(
    "# Temporal Graph Modeling for Illicit Transaction Detection\n"
    "### Elliptic Bitcoin Dataset & Walk-Forward Concept Drift\n\n"
    "**Objective:** Build a resilient anomaly detection architecture for the Elliptic Bitcoin dataset. We explore the impacts of graph structure, topological injection, directionality, and temporal modeling (LSTM vs Exponential Decay) in the presence of a massive structural regime shift (the Dark Market shutdown at **τ=43**)."
)

# ---------------------------------------------------------------- Slide 2: EDA A
md(
    "## 1. EDA Panel A — Class Imbalance\n"
    "The defining challenge of this dataset is the severe class imbalance. Only ~2% of labeled transactions are illicit."
)
code(
    "n_illicit = topo['N_illicit'].sum()\n"
    "n_licit = topo['N_licit'].sum()\n"
    "n_unknown = topo['N_unknown'].sum()\n"
    "labels = ['Illicit (2%)', 'Licit (21%)', 'Unknown (77%)']\n"
    "sizes = [n_illicit, n_licit, n_unknown]\n"
    "colors = ['#c0392b', '#2a9d4a', '#bdc3c7']\n"
    "explode = (0.2, 0, 0)\n"
    "fig, ax = plt.subplots(figsize=(6, 6))\n"
    "ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)\n"
    "ax.axis('equal')\n"
    "ax.set_title('Global Class Distribution')\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 3: EDA B
md(
    "## EDA Panel B — The Temporal Sequence\n"
    "The dataset represents a chronological sequence of 49 disconnected directed acyclic graphs (DAGs), each representing ~2 weeks of transactions. A massive market shock occurs at τ=43, collapsing the illicit volume."
)
code(
    "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
    "ax.plot(topo['Tau'], topo['N_nodes'], color='#34495e', lw=2, label='Total Nodes')\n"
    "ax.plot(topo['Tau'], topo['N_illicit'] * 10, color='#c0392b', lw=2, label='Illicit Nodes (x10)')\n"
    "ax.axvline(SHOCK, color='black', ls='--', lw=1.5, label='τ=43 Shock')\n"
    "ax.set_xlabel('snapshot τ')\n"
    "ax.set_ylabel('Count')\n"
    "ax.set_title('Transaction Volume per Snapshot')\n"
    "ax.legend()\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 4: EDA C
md(
    "## EDA Panel C — The Feature Hairball & PageRank\n"
    "**How to read this chart:**\n"
    "- **Top row (PCA):** Projects the 165 raw features to 2 dimensions. Notice the massive overlap between Licit (green) and Illicit (red). A linear tabular model struggles here.\n"
    "- **Middle row (t-SNE):** A non-linear projection. Still, we see a massive hairball without clear boundary separation.\n"
    "- **Bottom row (PageRank KDE):** When we examine topological metrics like PageRank, clear distributional shifts emerge. This proves that *structure* matters, motivating Graph Neural Networks."
)
code(
    "pca_df = pd.read_csv(os.path.join(RESULTS, 'eda_pca.csv'))\n"
    "tsne_df = pd.read_csv(os.path.join(RESULTS, 'eda_tsne.csv'))\n"
    "pr_df = pd.read_csv(os.path.join(RESULTS, 'eda_pagerank.csv'))\n"
    "fig = plt.figure(figsize=(20, 15))\n"
    "gs = fig.add_gridspec(3, 5)\n"
    "snapshots = [1, 42, 43, 44, 49]\n"
    "for i, tau in enumerate(snapshots):\n"
    "    ax_pca = fig.add_subplot(gs[0, i])\n"
    "    sub_pca = pca_df[pca_df['tau'] == tau]\n"
    "    licit_p = sub_pca[sub_pca['label'] == 0]\n"
    "    illicit_p = sub_pca[sub_pca['label'] == 1]\n"
    "    ax_pca.scatter(licit_p['pca1'], licit_p['pca2'], color='#2a9d4a', alpha=0.3, s=10, label='Licit')\n"
    "    ax_pca.scatter(illicit_p['pca1'], illicit_p['pca2'], color='#c0392b', alpha=0.8, s=15, label='Illicit')\n"
    "    ax_pca.set_title(f'PCA at τ={tau}')\n"
    "    if i == 0: ax_pca.legend()\n"
    "    \n"
    "    ax_tsne = fig.add_subplot(gs[1, i])\n"
    "    sub_tsne = tsne_df[tsne_df['tau'] == tau]\n"
    "    licit_t = sub_tsne[sub_tsne['label'] == 0]\n"
    "    illicit_t = sub_tsne[sub_tsne['label'] == 1]\n"
    "    ax_tsne.scatter(licit_t['tsne1'], licit_t['tsne2'], color='#2a9d4a', alpha=0.3, s=10)\n"
    "    ax_tsne.scatter(illicit_t['tsne1'], illicit_t['tsne2'], color='#c0392b', alpha=0.8, s=15)\n"
    "    ax_tsne.set_title(f'TSNE at τ={tau}')\n"
    "ax_pr = fig.add_subplot(gs[2, 1:4])\n"
    "sns.kdeplot(data=pr_df[pr_df['label']==0], x='pagerank', color='#2a9d4a', label='Licit', ax=ax_pr, log_scale=True)\n"
    "sns.kdeplot(data=pr_df[pr_df['label']==1], x='pagerank', color='#c0392b', label='Illicit', ax=ax_pr, log_scale=True)\n"
    "ax_pr.set_title('PageRank Distribution (Log Scale)')\n"
    "ax_pr.legend()\n"
    "fig.tight_layout()\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 5: Manifold
md(
    "## The Empirical Manifold Hypothesis\n"
    "By passing the raw features through $K$ steps of SGC propagation ($K=1 \\rightarrow 5$), we mix the neighborhood contexts. "
    "Here we've calculated the TwoNN intrinsic dimensionality for the **Undirected** configurations in our grid search immediately prior to the shock ($\tau=42$). We explicitly filter for `Directional=False` to isolate the core topological effects.\n\n"
    "The strong negative correlation visibly proves the manifold hypothesis: configurations that successfully compress the topological manifold (lower intrinsic dimension) lead directly to higher predictive accuracy (F1 Score)!"
)
code(
    "id_df = pd.read_csv(os.path.join(RESULTS, 'eda_grid_intrinsic_dim.csv'))\n"
    "id_df['Topo'] = id_df['Topo'].fillna('None')\n"
    "id_df = id_df[~id_df['Directional']]\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)\n"
    "sns.scatterplot(data=id_df[~id_df['PCA']], x='Intrinsic Dimension', y='F1 Score', hue='Topo', size='K', sizes=(50, 250), alpha=0.8, ax=axes[0])\n"
    "axes[0].set_title('Pre-PCA Configurations')\n"
    "axes[0].grid(True, alpha=0.3)\n"
    "axes[0].legend_.remove()\n"
    "sns.scatterplot(data=id_df[id_df['PCA']], x='Intrinsic Dimension', y='F1 Score', hue='Topo', size='K', sizes=(50, 250), alpha=0.8, ax=axes[1])\n"
    "axes[1].set_title('Post-PCA Configurations')\n"
    "axes[1].grid(True, alpha=0.3)\n"
    "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n"
    "fig.suptitle('Intrinsic Dimension vs F1 Score across Undirected SGC+MLP Configurations', fontsize=14)\n"
    "fig.tight_layout()\n"
    "plt.show()\n"
)

# ---------------------------------------------------------------- Slide 6: Baselines
md(
    "## 2. Weber Baselines & Establishing the Target\n"
    "Before building new architectures, we establish the static baselines provided by Weber et al.\n"
    "Logistic Regression provides a linear floor. GCN provides a deep graph reference. Random Forest and XGBoost completely dominate the static evaluation.\n\n"
    "**Why XGBoost?** While Random Forest scores slightly higher (0.80 vs 0.78), XGBoost offers vastly superior computational speed with highly comparable performance. Therefore, we select **XGBoost** as our primary baseline to beat moving forward."
)
code(
    "inst = [\n"
    "    ('Logistic Regression', 'Diagnostic: sklearn LR', '#95a5a6'),\n"
    "    ('GCN 2-layer', 'F3d: GCN reference [2-layer]', '#e67e22'),\n"
    "    ('Base XGBoost', 'F3a: Base XGBoost (clean)', '#1e8449'),\n"
    "    ('Random Forest', 'F3b: Random Forest (clean)', '#2a9d4a')\n"
    "]\n"
    "labels = [x[0] for x in inst]\n"
    "vals = [get_scalar(x[1], 'Static_OOT_F1') for x in inst]\n"
    "cols = [x[2] for x in inst]\n"
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "bars = ax.barh(labels, vals, color=cols)\n"
    "for b, v in zip(bars, vals):\n"
    "    ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center')\n"
    "ax.set_xlim(0, 0.95); ax.set_xlabel('Static OOT F1')\n"
    "ax.set_title('Static Performance: Weber Baselines')\n"
    "fig.tight_layout()\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 7: SGC Math
md(
    "## 3. The Mathematics of Simplified Graph Convolutions (SGC)\n"
    "Standard GCNs are non-linear between every propagation step, making them computationally heavy. "
    "We can collapse the weight matrices and apply a single linear propagation step:\n\n"
    "$$ \\mathbf{H}^{(K)} = \\tilde{\\mathbf{A}}^K \\mathbf{X} \\mathbf{\\Theta} $$\n\n"
    "However, illicit Bitcoin transactions exhibit **heterophily** (criminals transact with licit exchanges to launder money). "
    "To combat this, we implement **Multi-Scale SGC** with a non-linear MLP head:\n\n"
    "$$ \\mathbf{Z} = [\\mathbf{X}, \\tilde{\\mathbf{A}}\\mathbf{X}, \\tilde{\\mathbf{A}}^2\\mathbf{X}, \\dots, \\tilde{\\mathbf{A}}^K\\mathbf{X}] \\mathbf{W}_{MLP} $$\n\n"
    "By concatenating the $K$-hop features *before* passing them to a non-linear MLP, the network can explicitly learn heterogeneous graph shapes."
)

# ---------------------------------------------------------------- Slide 8: Grid Dir
md(
    "## 4. Grid Findings: Directionality\n"
    "Our grid search over graph directionality ($Dir=T$ vs $Dir=F$) revealed a fascinating oscillation based on K-depth:\n\n"
    "1. **Shallow ($K=1, 2$)**: Undirected wins. It safely pulls in immediate transaction partners without oversmoothing.\n"
    "2. **Medium ($K=3$)**: Directional *strictly* wins. In undirected propagation, a node's features bounce back and forth between itself and its neighbors, creating an echo chamber effect. Directional propagation strictly forces features downstream, acting as a **structural regularizer** against oversmoothing.\n"
)
code(
    "df = sweep.copy()\n"
    "df_grid = df[df['Sweep'].str.startswith('Grid: K=')].copy()\n"
    "df_grid['K'] = df_grid['Sweep'].str.extract(r'K=(\\d+)').astype(int)\n"
    "df_grid['Dir'] = df_grid['Sweep'].str.extract(r'Dir=([TF])')\n"
    "pivot = df_grid.pivot_table(index='K', columns='Dir', values='Static_OOT_F1_mean', aggfunc='max')\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "pivot.plot(kind='bar', color=['#2c6fbb', '#e67e22'], ax=ax)\n"
    "ax.set_ylabel('Max Static OOT F1'); ax.set_title('Undirected vs Directional performance by Propagation Depth')\n"
    "ax.legend(['Undirected (Dir=F)', 'Directional (Dir=T)'])\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 9: Grid Topo
md(
    "## 5. Grid Findings: Topology Injection & PCA\n"
    "**PCA Compression** (preserving 95% variance) drops raw dimensions from 165 $\\rightarrow$ 68. Mathematically, it strips away collinear noise before graph propagation.\n\n"
    "**Topology Injection** (PageRank, Degree):\n"
    "- **Early**: Concatenated to raw features *before* propagation (bleeds into neighbors).\n"
    "- **Late**: Concatenated *after* propagation (remains localized).\n"
    "**Finding**: At shallow depths ($K \le 2$), Early Injection rules. At deep depths, Topology injection becomes irrelevant, as SGC natively calculates global graph metrics!"
)

# ---------------------------------------------------------------- Slide 10: Grid Table
md(
    "## 6. Grid Analysis Table\n"
    "The complete statistical readout of the Grid Search (Static Evaluation). Notice the peak at K=5 Undirected."
)
code(
    "grid_df = df_grid[['Sweep', 'Variation', 'Static_OOT_F1_mean', 'Static_OOT_PRAUC_mean']].copy().dropna()\n"
    "grid_df.columns = ['Configuration', 'Variation', 'F1 Score', 'PR-AUC']\n"
    "grid_df['F1 Score'] = grid_df['F1 Score'].round(3)\n"
    "grid_df['PR-AUC'] = grid_df['PR-AUC'].round(3)\n"
    "grid_df = grid_df.sort_values(by='F1 Score', ascending=False).head(15)\n"
    "display(Markdown(grid_df.to_markdown(index=False)))\n"
)

# ---------------------------------------------------------------- Slide 11: K Reversal
md(
    "## 7. Walk-Forward Validation & The K-Depth Reversal\n"
    "When evaluated strictly on Static OOT, the massive $K=5$ Undirected model appears to be the champion. However, when subjected to rigorous **Walk-Forward Validation**, $K=5$ collapses. Its global undirected map is destroyed by the $\\tau=43$ concept drift. **$K=2$ becomes the true resilient champion by strictly modeling local peeling chains.**"
)
code(
    "wf_winners = [\n"
    "    ('K=1 Walk-Forward', 'F1: SGC+MLP WF K=1 [Dir=F; Topo=early; PCA]'),\n"
    "    ('K=2 Walk-Forward', 'F1: SGC+MLP WF K=2 [Dir=F; Topo=None]'),\n"
    "    ('K=3 Walk-Forward', 'F1: SGC+MLP WF K=3 [Dir=T; Topo=None; PCA]'),\n"
    "    ('K=5 Walk-Forward', 'F1: SGC+MLP WF K=5 [Dir=F; Topo=None; PCA]')\n"
    "]\n"
    "labels = [x[0] for x in wf_winners]\n"
    "vals_pre = [get_scalar(x[1], 'WF_Pre43_PRAUC') for x in wf_winners]\n"
    "vals_pooled = [get_scalar(x[1], 'WF_Pooled_F1') for x in wf_winners]\n"
    "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
    "x = np.arange(len(labels)); width = 0.35\n"
    "ax.bar(x - width/2, vals_pre, width, label='Pre-43 PRAUC', color='#e67e22')\n"
    "ax.bar(x + width/2, vals_pooled, width, label='Pooled F1', color='#8e44ad')\n"
    "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
    "ax.set_ylim(0, 1.0); ax.set_ylabel('Metric Score')\n"
    "ax.set_title('The K-Depth Reversal: K=2 Survives Walk-Forward Concept Drift')\n"
    "ax.legend()\n"
    "fig.tight_layout()\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 12: LSTM Math
md(
    "## 8. Testing the Sequence Model (LSTM)\n"
    "To capture temporal structure across the disconnected snapshots, we implemented an LSTM to propagate hidden states forward through time. "
    "Mathematically, the hidden state update is defined as:\n\n"
    "$$ h_t, c_t = \\text{LSTM}(x_t, (h_{t-1}, c_{t-1})) $$\n\n"
    "Where $x_t$ is a global graph representation (a mean-pooled broadcast vector of the entire SGC graph embedding at $\\tau=t$). "
    "The updated hidden state $h_t$ is concatenated to the node features to inform the MLP classifier."
)

# ---------------------------------------------------------------- Slide 13: LSTM Perf
md(
    "## 9. LSTM vs Static SGC+MLP Performance\n"
    "Does propagating this recurrent hidden state improve performance over the static graph model? Evaluating under strict walk-forward validation across the pre-shock window, we find that the SGC-LSTM actually underperforms the memoryless SGC+MLP. The sequence memory fails to beat the static baseline."
)
code(
    "order = ['SGC+MLP (Static)', 'SGC-LSTM']\n"
    "keys = ['F1: SGC+MLP WF K=2 [Dir=F; Topo=None]', 'F2: SGC-LSTM Chronological']\n"
    "vals = [get_scalar(k, 'WF_Pre43_PRAUC') for k in keys]\n"
    "fig, ax = plt.subplots(figsize=(6, 4))\n"
    "bars = ax.bar(order, vals, color=['#2c6fbb', '#d1495b'], width=0.5)\n"
    "for b, v in zip(bars, vals):\n"
    "    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center')\n"
    "ax.set_ylabel('Pre-Shock PR-AUC'); ax.set_ylim(0, 1.0)\n"
    "ax.set_title('Memory vs Memoryless Performance')\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 14: Decay Motivation
md(
    "## 10. Exponential Decay: Improving the Baseline XGBoost\n"
    "If global recurrence fails, how do we handle temporal concept drift? By targeting the loss function directly.\n\n"
    "Standard walk-forward validation treats all historical training data with uniform weight. This is detrimental under rapid concept drift (like the Dark Market shutdown). We introduce an Exponential Decay Sample Weight:\n\n"
    "$$ W = \lambda e^{-\lambda \Delta t} $$\n\n"
    "Where $\Delta t$ is the age of the snapshot. By applying this to our already powerful XGBoost baseline, we force the model to preferentially optimize for the most recent transactional patterns."
)

# ---------------------------------------------------------------- Slide 15: Per-Timestep Decay
md(
    "## 11. Per-Timestep Tracking: XGBoost Decay Recovery\n"
    "Let's observe how adjusting the decay rate $\lambda$ forces XGBoost to adapt across the $\\tau=43$ shock. While all models suffer a collapse immediately at the shock, the aggressive decay models recover predictive power far more effectively than the uniform baseline."
)
code(
    "f4_models = [\n"
    "    ('Uniform XGBoost', 'F1: Base XGBoost WF [v2]', '#2c6fbb'),\n"
    "    ('λ=0.05', 'F4: XGBoost decay λ=0.05', '#2a9d4a'),\n"
    "    ('λ=0.25', 'F4: XGBoost decay λ=0.25', '#f39c12'),\n"
    "    ('λ=0.50', 'F4: XGBoost decay λ=0.5', '#c0392b')\n"
    "]\n"
    "fig, ax = plt.subplots(figsize=(12, 6))\n"
    "for label, sweep_name, color in f4_models:\n"
    "    d = steps[steps['Sweep'] == sweep_name].sort_values('Tau')\n"
    "    if len(d) > 0:\n"
    "        ax.plot(d['Tau'], d['PRAUC'], marker='o', ms=4, color=color, label=label)\n"
    "ax.axvline(SHOCK, color='black', ls='--', lw=1.5)\n"
    "ax.axvspan(SHOCK - 0.5, 49.5, color='gray', alpha=0.10)\n"
    "ax.set_xlabel('Test Snapshot τ'); ax.set_ylabel('Per-step PR-AUC'); ax.set_ylim(0, 1.02)\n"
    "ax.set_title('Decay Tracking: XGBoost PR-AUC over time under concept drift')\n"
    "ax.legend()\n"
    "fig.tight_layout()\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 16: Ultimate Finding
md(
    "## 12. Our Strongest Finding: Exponential Decay Dominates\n"
    "When we aggregate the Walk-Forward Pooled F1 scores, the effect of Exponential Decay on the XGBoost baseline is immense. A gentle decay ($\lambda=0.05$) boosts the Pooled F1 from 0.826 to 0.871, establishing it as the undisputed champion architecture for Elliptic Bitcoin anomaly detection under regime shift."
)
code(
    "order = ['Uniform XGBoost', 'XGBoost λ=0.05', 'XGBoost λ=0.25', 'XGBoost λ=0.50']\n"
    "keys = ['F1: Base XGBoost WF [v2]', 'F4: XGBoost decay λ=0.05', 'F4: XGBoost decay λ=0.25', 'F4: XGBoost decay λ=0.5']\n"
    "vals = [get_scalar(k, 'WF_Pooled_F1') for k in keys]\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "bars = ax.bar(order, vals, color=['#2c6fbb', '#2a9d4a', '#f39c12', '#c0392b'])\n"
    "for b, v in zip(bars, vals):\n"
    "    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', weight='bold')\n"
    "ax.set_ylabel('Walk-Forward Pooled F1'); ax.set_ylim(0, 1.0)\n"
    "ax.set_title('Ultimate Finding: Exponential Decay massively improves Walk-Forward Performance')\n"
    "plt.show()"
)

# ---------------------------------------------------------------- Slide 17: Future Work
md(
    "## 13. Future Work & Discussion\n"
    "Our findings license several crucial avenues for future research:\n\n"
    "1. **Scalable Topological Features (Laplacian Centrality):**\n"
    "   - **The Idea:** Replace/augment raw in_degree and out_degree with Laplacian Centrality $C_L(v_i)$ to capture richer structural flow patterns (e.g., laundering peeling chains).\n"
    "   - **The Myth:** Sounds computationally prohibitive due to global graph energy calculations ($O(N^3)$).\n"
    "   - **The Reality:** Simplifies to a localized, 1-hop ego-network calculation:\n"
    "     $$C_L(v_i) = d_i^2 + d_i + 2 \\sum_{j \\in N(i)} d_j$$\n"
    "   - **The Efficiency:** Can be parallelized across all 200k nodes in milliseconds via a single sparse matrix multiplication ($O(|E|)$ complexity).\n"
    "2. **Synthetic Oversampling (SMOTE on Manifold):** We observed a severe 2% class imbalance. Given that SGC compresses the graph onto a smooth topological manifold, applying synthetic sampling (like SMOTE) directly in the SGC embedding space could radically improve minority class recall.\n"
    "3. **Per-Node Temporal Attention:** Our LSTM evaluated a *global* graph broadcast vector. Future architectures should explore per-node evolutionary weights (e.g., EvolveGCN) or Temporal Graph Attention Networks (TGAT) to track the explicit history of long-standing illicit addresses.\n"
    "4. **Dynamic Topology Construction:** Instead of relying purely on strict 2-week snapshots, constructing a continuous-time dynamic graph representation could prevent the boundary-cutoff issues inherent in discrete temporal modeling."
)

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
with open('presentation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('wrote presentation.ipynb with', len(cells), 'cells')
