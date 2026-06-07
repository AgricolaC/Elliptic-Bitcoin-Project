import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Synthetic dataset (same as Notebook 1 for continuity) ──────────────────
def make_dataset(n_normal=4000, n_anomaly=200, n_features=20, seed=SEED):
    rng = np.random.default_rng(seed)
    z   = rng.standard_normal((n_normal + 500, 2))
    z[:, 0] *= 3
    A = rng.standard_normal((2, n_features)) * 0.5
    X_normal  = z @ A + rng.standard_normal((n_normal + 500, n_features)) * 0.3
    X_anomaly = rng.standard_normal((n_anomaly, n_features)) * 4.0
    scaler = StandardScaler()
    X_all = scaler.fit_transform(np.vstack([X_normal, X_anomaly]))
    X_norm_s = X_all[:len(X_normal)].astype(np.float32)
    X_anom_s = X_all[len(X_normal):].astype(np.float32)
    X_train   = X_norm_s[:n_normal]
    X_test    = np.vstack([X_norm_s[n_normal:], X_anom_s])
    y_test    = np.array([0]*(len(X_norm_s)-n_normal) + [1]*n_anomaly)
    return X_train, X_test, y_test

X_train, X_test, y_test = make_dataset()
print(f'Train: {X_train.shape}  Test: {X_test.shape}  Anomaly rate: {y_test.mean():.1%}')

INPUT_DIM = 20
LATENT_DIM = 16

class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden*2),   nn.LeakyReLU(0.2),
            nn.Linear(hidden*2, out_dim),
        )
    def forward(self, z): return self.net(z)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden*2),  nn.LeakyReLU(0.2),
            nn.Linear(hidden*2, hidden),  nn.LeakyReLU(0.2),
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.features(x)
        return self.score(f), f


def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score, _ = critic(interp)
    grad = torch.autograd.grad(score, interp, torch.ones_like(score),
                                create_graph=True, retain_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(G, C, X_train, epochs=100, batch_size=256, lr=1e-4,
                  n_critic=5, gp_lambda=10, latent_dim=LATENT_DIM):
    G.to(DEVICE); C.to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    hist = {'C_loss': [], 'G_loss': []}
    for ep in range(epochs):
        c_losses, g_losses = [], []
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            # --- Train critic n_critic times ---
            for _ in range(n_critic):
                z = torch.randn(len(xb), latent_dim, device=DEVICE)
                fake = G(z).detach()
                c_real, _ = C(xb)
                c_fake, _ = C(fake)
                gp = gradient_penalty(C, xb, fake, DEVICE)
                c_loss = c_fake.mean() - c_real.mean() + gp_lambda * gp
                opt_C.zero_grad(); c_loss.backward(); opt_C.step()
                c_losses.append(c_loss.item())
            # --- Train generator ---
            z = torch.randn(len(xb), latent_dim, device=DEVICE)
            g_loss = -C(G(z))[0].mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            g_losses.append(g_loss.item())
        hist['C_loss'].append(np.mean(c_losses))
        hist['G_loss'].append(np.mean(g_losses))
        if (ep + 1) % 25 == 0:
            print(f'  Epoch {ep+1:3d}/{epochs}  C={hist["C_loss"][-1]:.3f}  G={hist["G_loss"][-1]:.3f}')
    return hist

G = Generator(LATENT_DIM, INPUT_DIM)
C = Critic(INPUT_DIM)
print('Training WGAN-GP on normal data...')
gan_hist = train_wgan_gp(G, C, X_train, epochs=100)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Training curves
ax = axes[0]
ep_range = range(1, len(gan_hist['C_loss'])+1)
ax.plot(ep_range, gan_hist['C_loss'], lw=1.5, color='steelblue', label='Critic loss')
ax.plot(ep_range, gan_hist['G_loss'], lw=1.5, color='tomato',    label='Generator loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('WGAN-GP training dynamics', fontweight='bold')
ax.legend()

# Quality check: real vs generated distributions (PCA)
ax = axes[1]
G.eval()
with torch.no_grad():
    z_sample = torch.randn(1000, LATENT_DIM, device=DEVICE)
    X_fake   = G(z_sample).cpu().numpy()

pca = PCA(n_components=2)
both_2d = pca.fit_transform(np.vstack([X_train[:1000], X_fake]))
real_2d = both_2d[:1000]
fake_2d = both_2d[1000:]

ax.scatter(*real_2d.T, s=5, alpha=0.3, color='steelblue', label='Real (normal)')
ax.scatter(*fake_2d.T, s=5, alpha=0.3, color='tomato',    label='Generated')
ax.set_title('PCA: Real vs Generated samples', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/wgan_training.png', bbox_inches='tight')
plt.show()

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden*2),  nn.LeakyReLU(0.2),
            nn.Linear(hidden*2, hidden),  nn.LeakyReLU(0.2),
            nn.Linear(hidden, latent_dim),
        )
    def forward(self, x): return self.net(x)


def train_encoder_fanogan(E, G_frozen, C_frozen, X_train,
                          epochs=80, batch_size=256, lr=1e-4, kappa=1.0):
    """f-AnoGAN encoder training: image + feature matching loss."""
    E.to(DEVICE)
    G_frozen.eval(); C_frozen.eval()
    for p in G_frozen.parameters(): p.requires_grad_(False)
    for p in C_frozen.parameters(): p.requires_grad_(False)

    opt = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    loader = DataLoader(TensorDataset(torch.tensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            z_hat = E(xb)
            x_rec = G_frozen(z_hat)

            # Image distance
            loss_img = nn.MSELoss()(x_rec, xb)

            # Feature matching on discriminator intermediate features
            _, feat_real = C_frozen(xb)
            _, feat_rec  = C_frozen(x_rec)
            loss_feat = nn.MSELoss()(feat_rec, feat_real.detach())

            loss = loss_img + kappa * loss_feat
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        losses.append(ep_loss / len(X_train))
        if (ep + 1) % 20 == 0:
            print(f'  Epoch {ep+1:3d}/{epochs}  loss={losses[-1]:.5f}')
    return losses

E = Encoder(INPUT_DIM, LATENT_DIM)
print('Training f-AnoGAN encoder...')
enc_losses = train_encoder_fanogan(E, G, C, X_train)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(enc_losses, color='darkorange', lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('f-AnoGAN encoder training', fontweight='bold')
plt.tight_layout(); plt.show()

def fanogan_scores(E, G, C, X, kappa=1.0, batch_size=256):
    E.eval(); G.eval(); C.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size]).to(DEVICE)
            z_hat = E(xb)
            x_rec = G(z_hat)
            img_dist  = ((xb - x_rec) ** 2).mean(dim=1)
            _, f_real = C(xb)
            _, f_rec  = C(x_rec)
            feat_dist = ((f_real - f_rec) ** 2).mean(dim=1)
            all_scores.append((img_dist + kappa * feat_dist).cpu().numpy())
    return np.concatenate(all_scores)

scores_fanogan = fanogan_scores(E, G, C, X_test)
auc_gan = roc_auc_score(y_test, scores_fanogan)
ap_gan  = average_precision_score(y_test, scores_fanogan)
print(f'f-AnoGAN  AUC-ROC: {auc_gan:.4f}   AP: {ap_gan:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Score distribution
ax = axes[0]
ax.hist(scores_fanogan[y_test==0], bins=60, alpha=0.6, density=True, color='steelblue', label='Normal')
ax.hist(scores_fanogan[y_test==1], bins=60, alpha=0.6, density=True, color='tomato',   label='Anomaly')
ax.set_xlabel('f-AnoGAN score'); ax.set_ylabel('Density')
ax.set_title('f-AnoGAN score distribution', fontweight='bold')
ax.legend()

# ROC
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, scores_fanogan)
ax.plot(fpr, tpr, lw=2, color='darkorange', label=f'f-AnoGAN (AUC={auc_gan:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC curve', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/fanogan_scores.png', bbox_inches='tight')
plt.show()

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA as skPCA

# --- Classical baselines ---
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED)
iso.fit(X_train)
scores_iso = -iso.score_samples(X_test)

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')
ocsvm.fit(X_train[:1000])  # subsample for speed
scores_ocsvm = -ocsvm.score_samples(X_test)

# PCA reconstruction error
pca_det = skPCA(n_components=5)
pca_det.fit(X_train)
X_test_pca = pca_det.inverse_transform(pca_det.transform(X_test))
scores_pca = ((X_test - X_test_pca) ** 2).mean(axis=1)

# z-score max per sample
mu_tr = X_train.mean(0); std_tr = X_train.std(0)
scores_zscore = np.abs((X_test - mu_tr) / (std_tr + 1e-9)).max(axis=1)

# Vanilla AE (reload from NB1 approach, train fresh here)
class VanillaAE_simple(nn.Module):
    def __init__(self, d=20, z=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,z))
        self.dec = nn.Sequential(nn.Linear(z,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,d))
    def forward(self,x): return self.dec(self.enc(x))

ae2 = VanillaAE_simple().to(DEVICE)
opt_ae = optim.Adam(ae2.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=256, shuffle=True)
for _ in range(60):
    for (xb,) in loader:
        xb=xb.to(DEVICE); l=nn.MSELoss()(ae2(xb),xb)
        opt_ae.zero_grad(); l.backward(); opt_ae.step()
ae2.eval()
with torch.no_grad():
    xt=torch.tensor(X_test).to(DEVICE)
    scores_ae2 = ((xt - ae2(xt))**2).mean(dim=1).cpu().numpy()

# Compile all results
methods = {
    'z-score (max)':    scores_zscore,
    'PCA recon':        scores_pca,
    'Isolation Forest': scores_iso,
    'One-Class SVM':    scores_ocsvm,
    'Vanilla AE':       scores_ae2,
    'f-AnoGAN':         scores_fanogan,
}

print(f'  {"Method":<22} {"AUC-ROC":>9} {"Avg Prec":>10}')
print('  ' + '-'*43)
method_aucs = {}
for name, sc in methods.items():
    a = roc_auc_score(y_test, sc)
    p = average_precision_score(y_test, sc)
    method_aucs[name] = (a, p)
    print(f'  {name:<22} {a:>9.4f} {p:>10.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curves all methods
ax = axes[0]
colors_m = ['#888', '#4CAF50', '#2196F3', '#9C27B0', 'darkorange', 'tomato']
for (name, sc), col in zip(methods.items(), colors_m):
    fpr, tpr, _ = roc_curve(y_test, sc)
    auc = roc_auc_score(y_test, sc)
    ax.plot(fpr, tpr, lw=1.8, color=col, label=f'{name} ({auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC curves — all Module 2 methods', fontweight='bold')
ax.legend(fontsize=8)

# Bar chart AUC-ROC
ax = axes[1]
names = list(method_aucs.keys())
aucs  = [v[0] for v in method_aucs.values()]
bars = ax.barh(names, aucs, color=colors_m, alpha=0.75)
ax.axvline(0.5, color='gray', ls='--', lw=1, alpha=0.5)
for bar, a in zip(bars, aucs):
    ax.text(a + 0.005, bar.get_y() + bar.get_height()/2,
            f'{a:.3f}', va='center', fontsize=9)
ax.set_xlim(0.4, 1.05)
ax.set_xlabel('AUC-ROC')
ax.set_title('Summary: AUC-ROC by method', fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/method_comparison.png', bbox_inches='tight')
plt.show()

def build_graph_with_anomalies(n_normal=200, n_anom=20, seed=SEED):
    """
    Normal nodes: Barabasi-Albert graph (power-law degrees, realistic).
    Anomaly types injected:
      - Dense clique (subgraph anomaly)
      - Hub nodes with very high degree
      - Isolated stars (spoke-and-hub)
    """
    rng = np.random.default_rng(seed)
    # Base graph
    G = nx.barabasi_albert_graph(n_normal, 3, seed=seed)
    labels = {v: 0 for v in G.nodes()}

    n = n_normal
    # Anomaly 1: dense clique
    clique_size = 8
    clique_nodes = list(range(n, n + clique_size))
    G.add_nodes_from(clique_nodes)
    for u in clique_nodes:
        for v in clique_nodes:
            if u < v:
                G.add_edge(u, v)
        # Connect clique to a random normal node
        G.add_edge(u, rng.integers(0, n_normal))
    for u in clique_nodes:
        labels[u] = 1
    n += clique_size

    # Anomaly 2: super-hub (connects to many normal nodes)
    hub_nodes = list(range(n, n + 3))
    G.add_nodes_from(hub_nodes)
    for hub in hub_nodes:
        targets = rng.choice(n_normal, size=rng.integers(30, 50), replace=False)
        for t in targets:
            G.add_edge(hub, t)
        labels[hub] = 1
    n += 3

    # Anomaly 3: isolated star (one centre, many leaves, minimal connection to rest)
    star_center = n
    star_leaves  = list(range(n+1, n+6))
    G.add_node(star_center)
    G.add_nodes_from(star_leaves)
    for leaf in star_leaves:
        G.add_edge(star_center, leaf)
    G.add_edge(star_center, rng.integers(0, n_normal))  # one link to main graph
    labels[star_center] = 1
    for leaf in star_leaves:
        labels[leaf] = 1

    node_labels = np.array([labels[v] for v in G.nodes()])
    return G, node_labels

G_graph, node_labels = build_graph_with_anomalies()
print(f'Nodes: {G_graph.number_of_nodes()}   Edges: {G_graph.number_of_edges()}')
print(f'Anomalous nodes: {node_labels.sum()}  ({node_labels.mean():.1%})')

# Visualise the graph
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Use spring layout, computed once
pos = nx.spring_layout(G_graph, seed=SEED, k=0.4)
node_colors = ['tomato' if node_labels[i] else 'steelblue'
               for i in range(len(node_labels))]
node_sizes  = [60 if node_labels[i] else 15
               for i in range(len(node_labels))]

ax = axes[0]
ax.set_title('Graph with anomalous nodes (red)', fontweight='bold')
nx.draw_networkx_nodes(G_graph, pos, ax=ax, node_color=node_colors,
                       node_size=node_sizes, alpha=0.8)
nx.draw_networkx_edges(G_graph, pos, ax=ax, alpha=0.12, width=0.5)
normal_patch = plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor='steelblue', markersize=8, label='Normal')
anom_patch   = plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor='tomato',   markersize=10, label='Anomaly')
ax.legend(handles=[normal_patch, anom_patch], loc='upper left')
ax.axis('off')

# Degree distribution
ax = axes[1]
degrees_normal = [G_graph.degree(v) for v in G_graph.nodes() if node_labels[list(G_graph.nodes()).index(v)] == 0]
degrees_anom   = [G_graph.degree(v) for v in G_graph.nodes() if node_labels[list(G_graph.nodes()).index(v)] == 1]
ax.hist(degrees_normal, bins=30, alpha=0.6, color='steelblue', density=True, label='Normal')
ax.hist(degrees_anom,   bins=20, alpha=0.7, color='tomato',   density=True, label='Anomaly')
ax.set_xlabel('Node degree'); ax.set_ylabel('Density')
ax.set_title('Degree distribution: normal vs anomalous', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/graph_overview.png', bbox_inches='tight')
plt.show()

def extract_node_features(G):
    """Compute a feature vector for each node."""
    nodes = list(G.nodes())
    deg   = dict(G.degree())
    clust = nx.clustering(G)

    # Average neighbour degree
    avg_nbr_deg = nx.average_neighbor_degree(G)

    # Ego-net statistics
    features = []
    for v in nodes:
        ego = nx.ego_graph(G, v, radius=1)
        ego_nodes  = ego.number_of_nodes()
        ego_edges  = ego.number_of_edges()
        ego_density = nx.density(ego)

        # Triangle count
        triangles = nx.triangles(G, v)

        # OddBall: deviation from power law
        # Expected edges ~ nodes^1.5
        expected_edges = ego_nodes ** 1.5
        oddball = abs(ego_edges - expected_edges) / (expected_edges + 1)

        features.append([
            deg[v],                     # degree
            clust[v],                   # clustering coefficient
            avg_nbr_deg[v],             # avg neighbour degree
            ego_nodes,                  # ego-net size
            ego_edges,                  # ego-net edges
            ego_density,                # ego-net density
            triangles,                  # triangle count
            oddball,                    # OddBall score
        ])
    return np.array(features, dtype=np.float32), nodes

print('Extracting node features (may take ~30 sec for large graphs)...')
X_nodes, node_ids = extract_node_features(G_graph)
print(f'Feature matrix: {X_nodes.shape}')
feature_names = ['Degree','Clustering','Avg nbr deg','Ego nodes',
                 'Ego edges','Ego density','Triangles','OddBall']

# Isolation Forest on node features
iso_graph = IsolationForest(n_estimators=200, contamination=0.10, random_state=SEED)
iso_graph.fit(X_nodes[node_labels == 0])
scores_graph = -iso_graph.score_samples(X_nodes)

auc_graph = roc_auc_score(node_labels, scores_graph)
print(f'\nIsolation Forest on node features  AUC-ROC: {auc_graph:.4f}')

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for idx, (fname, ax) in enumerate(zip(feature_names, axes.flat)):
    ax.hist(X_nodes[node_labels==0, idx], bins=40, alpha=0.6, density=True,
            color='steelblue', label='Normal')
    ax.hist(X_nodes[node_labels==1, idx], bins=20, alpha=0.7, density=True,
            color='tomato', label='Anomaly')
    auc_f = roc_auc_score(node_labels, X_nodes[:, idx])
    ax.set_title(f'{fname}\n(AUC={auc_f:.3f})', fontsize=9, fontweight='bold')
    if idx % 4 == 0:
        ax.set_ylabel('Density')

handles = [mpatches.Patch(color='steelblue', alpha=0.6, label='Normal'),
           mpatches.Patch(color='tomato',    alpha=0.7, label='Anomaly')]
fig.legend(handles=handles, loc='upper right', fontsize=9)
plt.suptitle('Node feature distributions: normal vs anomalous', fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/graph_features.png', bbox_inches='tight')
plt.show()

import matplotlib.patches as mpatches

# Build normalised Laplacian
A_mat  = nx.to_scipy_sparse_array(G_graph, format='csr', dtype=float)
degree_vec = np.array(A_mat.sum(axis=1)).flatten()
degree_vec = np.maximum(degree_vec, 1e-9)  # avoid division by zero
D_invsqrt  = sparse.diags(1.0 / np.sqrt(degree_vec))
L_sym = sparse.eye(G_graph.number_of_nodes()) - D_invsqrt @ A_mat @ D_invsqrt

# Compute first k eigenvectors (smallest eigenvalues)
k = 16
eigenvalues, eigenvectors = eigsh(L_sym, k=k, which='SM')  # SM = smallest magnitude
# Sort by eigenvalue
idx_sort = np.argsort(eigenvalues)
eigenvalues  = eigenvalues[idx_sort]
eigenvectors = eigenvectors[:, idx_sort]

print(f'First {k} eigenvalues: {eigenvalues[:8].round(4)}')

# Spectral embedding (skip constant eigenvector at λ=0)
spectral_emb = eigenvectors[:, 1:]   # (N, k-1)

# Anomaly score: reconstruction error in spectral space
pca_spectral = skPCA(n_components=4)
spectral_reduced = pca_spectral.fit_transform(spectral_emb)
spectral_recon   = pca_spectral.inverse_transform(spectral_reduced)
scores_spectral  = ((spectral_emb - spectral_recon) ** 2).mean(axis=1)

# Also: Isolation Forest in spectral embedding
iso_spectral = IsolationForest(n_estimators=200, contamination=0.10, random_state=SEED)
iso_spectral.fit(spectral_emb[node_labels==0])
scores_spectral_iso = -iso_spectral.score_samples(spectral_emb)

auc_sp      = roc_auc_score(node_labels, scores_spectral)
auc_sp_iso  = roc_auc_score(node_labels, scores_spectral_iso)
print(f'Spectral recon error    AUC-ROC: {auc_sp:.4f}')
print(f'Spectral + Iso Forest   AUC-ROC: {auc_sp_iso:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Spectral embedding (first 2 non-trivial eigenvectors)
ax = axes[0]
ax.scatter(spectral_emb[node_labels==0, 0], spectral_emb[node_labels==0, 1],
           s=10, alpha=0.5, c='steelblue', label='Normal')
ax.scatter(spectral_emb[node_labels==1, 0], spectral_emb[node_labels==1, 1],
           s=40, alpha=0.9, c='tomato', label='Anomaly', marker='x', linewidths=1.5)
ax.set_xlabel('Eigenvector 2 (Fiedler)')
ax.set_ylabel('Eigenvector 3')
ax.set_title('Spectral embedding (2nd and 3rd eigenvectors)', fontweight='bold')
ax.legend()

# Eigenspectrum
ax = axes[1]
ax.stem(range(k), eigenvalues, linefmt='steelblue', markerfmt='o', basefmt='gray')
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Eigenvalue λ')
ax.set_title('Normalised Laplacian eigenspectrum', fontweight='bold')
ax.axhline(1, color='gray', ls='--', lw=1, alpha=0.5, label='λ=1 (boundary)')
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/spectral_anomaly.png', bbox_inches='tight')
plt.show()

class GCNLayer(nn.Module):
    """Simple GCN layer: H' = ReLU( D^{-1/2} A D^{-1/2} H W )"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, H, A_norm):  # A_norm: dense normalised adjacency
        return self.act(A_norm @ self.W(H))


class DOMINANT(nn.Module):
    def __init__(self, in_dim, hidden=32, embed=16, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, embed)
        self.attr_dec = nn.Linear(embed, in_dim)

    def forward(self, X, A_norm):
        H1 = self.gcn1(X, A_norm)
        Z  = self.gcn2(H1, A_norm)
        # Attribute reconstruction
        X_hat = self.attr_dec(Z)
        # Structure reconstruction: Â = sigmoid(Z Zᵀ)
        A_hat = torch.sigmoid(Z @ Z.t())
        return X_hat, A_hat, Z

    def anomaly_score(self, X, A_norm, A_true):
        with torch.no_grad():
            X_hat, A_hat, _ = self.forward(X, A_norm)
            attr_err   = ((X - X_hat) ** 2).mean(dim=1)
            struct_err = ((A_true - A_hat) ** 2).mean(dim=1)
            return (self.alpha * attr_err + (1 - self.alpha) * struct_err).cpu().numpy()


def build_dominant_inputs(G, node_feats):
    """Build normalised adjacency matrix and node feature tensor."""
    A = nx.to_numpy_array(G)
    A = A + np.eye(len(A))  # add self-loops
    D = np.diag(A.sum(axis=1) ** -0.5)
    A_norm = D @ A @ D
    A_t    = torch.tensor(A_norm, dtype=torch.float32)
    A_true = torch.tensor(A,      dtype=torch.float32)
    X_t    = torch.tensor(node_feats, dtype=torch.float32)
    # Normalise features
    X_t = (X_t - X_t.mean(0)) / (X_t.std(0) + 1e-9)
    return X_t, A_t, A_true


def train_dominant(model, X_t, A_t, A_true, epochs=200, lr=1e-3, alpha=0.5):
    model.to(DEVICE)
    X_t, A_t, A_true = X_t.to(DEVICE), A_t.to(DEVICE), A_true.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        X_hat, A_hat, _ = model(X_t, A_t)
        loss_attr   = nn.MSELoss()(X_hat, X_t)
        loss_struct = nn.BCELoss()(A_hat, A_true)
        loss = alpha * loss_attr + (1 - alpha) * loss_struct
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (ep+1) % 50 == 0:
            print(f'  Epoch {ep+1:3d}/{epochs}  loss={losses[-1]:.5f}')
    return losses


X_g, A_norm_g, A_true_g = build_dominant_inputs(G_graph, X_nodes)
dom_model = DOMINANT(in_dim=X_nodes.shape[1], hidden=32, embed=16)
print('Training DOMINANT...')
dom_losses = train_dominant(dom_model, X_g, A_norm_g, A_true_g)

dom_model.eval()
dom_scores = dom_model.anomaly_score(
    X_g.to(DEVICE), A_norm_g.to(DEVICE), A_true_g.to(DEVICE)
)
auc_dom = roc_auc_score(node_labels, dom_scores)
print(f'\nDOMINANT  AUC-ROC: {auc_dom:.4f}')

# Final comparison of all graph methods
graph_methods = {
    'Isolation Forest (features)': scores_graph,
    'Spectral + Iso Forest':        scores_spectral_iso,
    'Spectral recon error':          scores_spectral,
    'DOMINANT (GCN-AE)':             dom_scores,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_g = ['#2196F3', '#4CAF50', '#9C27B0', 'tomato']

ax = axes[0]
for (name, sc), col in zip(graph_methods.items(), colors_g):
    fpr, tpr, _ = roc_curve(node_labels, sc)
    auc = roc_auc_score(node_labels, sc)
    ax.plot(fpr, tpr, lw=2, color=col, label=f'{name} ({auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC — graph anomaly methods', fontweight='bold')
ax.legend(fontsize=8)

# Visualise DOMINANT scores on graph
ax = axes[1]
ax.set_title('DOMINANT anomaly scores on graph', fontweight='bold')
dom_norm = (dom_scores - dom_scores.min()) / (dom_scores.max() - dom_scores.min() + 1e-9)
nx.draw_networkx_nodes(G_graph, pos, ax=ax,
                       node_color=dom_norm, cmap='RdYlBu_r',
                       node_size=[30+50*s for s in dom_norm], alpha=0.85)
nx.draw_networkx_edges(G_graph, pos, ax=ax, alpha=0.08, width=0.4)
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                            norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Normalised anomaly score')
ax.axis('off')

plt.tight_layout()
plt.savefig('/tmp/graph_comparison.png', bbox_inches='tight')
plt.show()
