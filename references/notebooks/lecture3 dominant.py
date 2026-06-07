# Colab-friendly install. Safe to re-run.
import sys, subprocess

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

try:
    import torch_geometric  # noqa: F401
except ImportError:
    # PyG ships wheels matching the local torch version; the meta-package handles it.
    pip_install("torch_geometric")
    import torch_geometric  # noqa: F401

print("torch_geometric", torch_geometric.__version__)


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, degree
from torch_geometric.nn import GCNConv

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("torch:", torch.__version__)


dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0]

n = data.num_nodes
d = data.num_features
m = data.num_edges // 2  # PyG stores both directions

print(f"|V| = {n} nodes")
print(f"|E| = {m} undirected edges")
print(f"features X in R^(n x {d})")
print(f"avg degree = {2*m/n:.2f}")


def inject_anomalies(
    data,
    n_cliques=20,
    clique_size=15,
    n_attr_anomalies=300,
    k_candidates=50,
    seed=SEED,
):
    '''Inject structural (clique) and contextual (attribute) anomalies into a PyG Data object.

    Returns a new Data object with the corrupted edge_index and x, plus a
    boolean tensor `y_anom` of shape [n] marking the injected anomalies.
    '''
    rng = np.random.default_rng(seed)
    n = data.num_nodes
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    y_anom = torch.zeros(n, dtype=torch.bool)

    # --- 3.1 Structural anomalies: planted cliques ----------------------
    all_nodes = rng.permutation(n)
    pool = all_nodes[: n_cliques * clique_size]
    new_edges = []
    for c in range(n_cliques):
        members = pool[c * clique_size : (c + 1) * clique_size]
        # full pairwise undirected edges
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                u, v = int(members[i]), int(members[j])
                new_edges.append((u, v))
                new_edges.append((v, u))
        y_anom[members] = True

    if new_edges:
        extra = torch.tensor(new_edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, extra], dim=1)

    # --- 3.2 Contextual anomalies: attribute perturbation ---------------
    # Pick nodes NOT already anomalous to flip the attributes of.
    normal = np.where(~y_anom.numpy())[0]
    attr_targets = rng.choice(normal, size=n_attr_anomalies, replace=False)

    for v in attr_targets:
        cand = rng.choice(n, size=k_candidates, replace=False)
        # farthest-feature candidate (in L2)
        dists = torch.norm(x[cand] - x[v], dim=1)
        winner = cand[int(torch.argmax(dists))]
        x[v] = x[winner]
        y_anom[v] = True

    # Build the new Data object
    new_data = data.clone()
    new_data.x = x
    new_data.edge_index = edge_index
    new_data.y_anom = y_anom
    return new_data


data_anom = inject_anomalies(data)

n_total = data_anom.num_nodes
n_inj = int(data_anom.y_anom.sum().item())
print(f"injected anomalies: {n_inj} / {n_total}  ({100*n_inj/n_total:.2f}%)")
print(f"new edge count: {data_anom.num_edges // 2} (vs original {data.num_edges // 2})")


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# --- degree distribution by class -----------------------------------
deg = degree(data_anom.edge_index[0], num_nodes=data_anom.num_nodes).cpu().numpy()
is_anom = data_anom.y_anom.numpy()

axes[0].hist(deg[~is_anom], bins=40, alpha=0.6, label="normal", color="#1565c0")
axes[0].hist(deg[is_anom],  bins=40, alpha=0.6, label="anomalous", color="#c62828")
axes[0].set_xlabel("node degree")
axes[0].set_ylabel("count")
axes[0].set_title("Degree distribution")
axes[0].set_yscale("log")
axes[0].legend()

# --- 2-D PCA projection of features ---------------------------------
from sklearn.decomposition import PCA
proj = PCA(n_components=2, random_state=SEED).fit_transform(data_anom.x.numpy())
axes[1].scatter(proj[~is_anom, 0], proj[~is_anom, 1], s=4, alpha=0.4,
                color="#1565c0", label="normal")
axes[1].scatter(proj[is_anom, 0], proj[is_anom, 1], s=10, alpha=0.8,
                color="#c62828", label="anomalous")
axes[1].set_title("PCA of node features X")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].legend()

plt.tight_layout(); plt.show()


def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    '''Return  hat_A = D^{-1/2} (A + I) D^{-1/2}  as a dense tensor.

    Args
    ----
    A : (n, n) binary or weighted adjacency, assumed symmetric.

    Returns
    -------
    hat_A : (n, n) dense, symmetric, with spectrum in [-1, 1].
    '''
    n = A.size(0)
    A_tilde = A + torch.eye(n, device=A.device)
    d_tilde = A_tilde.sum(dim=1)
    d_inv_sqrt = torch.pow(d_tilde, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


class GCNLayer(nn.Module):
    '''Dense GCN layer following Kipf & Welling (2017).

    H_{l+1} = activation( hat_A @ H_l @ W_l )

    Notes
    -----
    Dense to keep the code as close to the lecture formula as possible.
    For Cora (n=2708) the dense matrices are tiny and dense ops beat sparse
    ones in PyTorch on most hardware.
    '''
    def __init__(self, in_dim: int, out_dim: int, activation=F.relu):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation = activation

    def forward(self, H: torch.Tensor, hat_A: torch.Tensor) -> torch.Tensor:
        out = hat_A @ H @ self.W
        return out if self.activation is None else self.activation(out)


# Build the dense adjacency for Cora
A_dense = to_dense_adj(data_anom.edge_index, max_num_nodes=data_anom.num_nodes)[0]
hat_A = normalize_adj(A_dense).to(device)
X = data_anom.x.to(device)

# Our layer
ours = GCNLayer(data_anom.num_features, 32, activation=None).to(device)

# PyG layer, with bias disabled and the SAME weight matrix
theirs = GCNConv(data_anom.num_features, 32, bias=False, add_self_loops=True,
                 normalize=True).to(device)
with torch.no_grad():
    # PyG stores the linear weight as (out, in); ours as (in, out)
    theirs.lin.weight.copy_(ours.W.t())

with torch.no_grad():
    out_ours   = ours(X, hat_A)
    out_theirs = theirs(X, data_anom.edge_index.to(device))

diff = (out_ours - out_theirs).abs().max().item()
print(f"max |ours - PyG| = {diff:.2e}")
assert diff < 1e-4, "GCN layer disagrees with PyG — check normalisation."
print("OK: our GCN matches PyG.")


class DOMINANT(nn.Module):
    '''DOMINANT (Ding et al., SDM 2019).

    Encoder:   2-layer GCN producing embeddings Z in R^{n x h}.
    Decoders:  (a) attribute: 1-layer GCN -> X_hat  in R^{n x d}
               (b) structure: inner product -> A_hat in (0,1)^{n x n}
    '''

    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.enc1 = GCNLayer(in_dim, hidden_dim)
        self.enc2 = GCNLayer(hidden_dim, embed_dim)
        self.attr_dec = GCNLayer(embed_dim, in_dim, activation=None)
        self.dropout = dropout

    def encode(self, X, hat_A):
        h = self.enc1(X, hat_A)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.enc2(h, hat_A)
        return z

    def decode_attr(self, Z, hat_A):
        return self.attr_dec(Z, hat_A)

    @staticmethod
    def decode_struct(Z):
        # Inner product decoder, A_hat = sigmoid(Z Z^T).
        return torch.sigmoid(Z @ Z.t())

    def forward(self, X, hat_A):
        Z = self.encode(X, hat_A)
        X_hat = self.decode_attr(Z, hat_A)
        A_hat = self.decode_struct(Z)
        return X_hat, A_hat, Z


def train_dominant(
    data_anom,
    hat_A,
    hidden_dim=64,
    embed_dim=32,
    alpha=0.5,
    lr=5e-3,
    weight_decay=0.0,
    n_epochs=200,
    verbose=True,
):
    '''Train DOMINANT and return the trained model and the training-loss curve.'''
    X = data_anom.x.to(device)
    A_dense = to_dense_adj(data_anom.edge_index,
                           max_num_nodes=data_anom.num_nodes)[0].to(device)

    model = DOMINANT(data_anom.num_features, hidden_dim, embed_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        optim.zero_grad()
        X_hat, A_hat, _ = model(X, hat_A)
        # Frobenius norms, averaged over n
        loss_attr   = ((X - X_hat) ** 2).sum() / data_anom.num_nodes
        loss_struct = ((A_dense - A_hat) ** 2).sum() / data_anom.num_nodes
        loss = alpha * loss_attr + (1 - alpha) * loss_struct
        loss.backward()
        optim.step()
        history.append((loss.item(), loss_attr.item(), loss_struct.item()))
        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  "
                  f"attr={loss_attr.item():.4f}  struct={loss_struct.item():.4f}")
    return model, history


model, history = train_dominant(data_anom, hat_A, alpha=0.5, n_epochs=200)


hist = np.array(history)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(hist[:, 0], label="total", color="#0f3460", lw=2)
ax.plot(hist[:, 1], label="attribute", color="#118ab2", lw=1, ls="--")
ax.plot(hist[:, 2], label="structure", color="#c62828", lw=1, ls="--")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.set_title("Training loss"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def score_dominant(model, X, hat_A, A_dense, alpha=0.5):
    '''Per-node DOMINANT anomaly score.'''
    model.eval()
    X_hat, A_hat, _ = model(X, hat_A)
    err_attr   = torch.norm(X - X_hat, dim=1)        # (n,)
    err_struct = torch.norm(A_dense - A_hat, dim=1)  # (n,)
    s = alpha * err_attr + (1 - alpha) * err_struct
    return s.cpu().numpy(), err_attr.cpu().numpy(), err_struct.cpu().numpy()


def precision_at_k(y_true, scores, k):
    '''Fraction of the top-k scored items that are anomalous.'''
    top_k = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k].sum()) / k


def report(name, y_true, scores, k=None):
    if k is None:
        k = int(y_true.sum())
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    pak   = precision_at_k(y_true, scores, k)
    print(f"{name:>22s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  P@{k}={pak:.4f}")
    return auroc, auprc, pak


X = data_anom.x.to(device)
A_dense = to_dense_adj(data_anom.edge_index,
                       max_num_nodes=data_anom.num_nodes)[0].to(device)
y_true = data_anom.y_anom.numpy()

scores, e_attr, e_struct = score_dominant(model, X, hat_A, A_dense, alpha=0.5)
report("DOMINANT (alpha=0.5)", y_true, scores)
report("attribute only",       y_true, e_attr)
report("structure only",       y_true, e_struct)


fig, ax = plt.subplots(figsize=(7, 3.6))
bins = np.linspace(scores.min(), scores.max(), 60)
ax.hist(scores[~y_true], bins=bins, alpha=0.6, label="normal", color="#1565c0")
ax.hist(scores[ y_true], bins=bins, alpha=0.7, label="anomalous", color="#c62828")
ax.set_xlabel("DOMINANT anomaly score s(v)")
ax.set_ylabel("count")
ax.set_yscale("log")
ax.set_title("Score distribution: normal vs anomalous nodes")
ax.legend()
plt.tight_layout(); plt.show()


import networkx as nx
from sklearn.ensemble import IsolationForest

# Build a NetworkX graph for the structural features
G = nx.Graph()
G.add_nodes_from(range(data_anom.num_nodes))
edges = data_anom.edge_index.t().numpy()
G.add_edges_from(edges)

deg_dict = dict(G.degree())
clust    = nx.clustering(G)
# avg neighbour degree
avg_nbr_deg = nx.average_neighbor_degree(G)
# ego-net edge count
ego_edges = {v: G.subgraph(list(G.neighbors(v)) + [v]).number_of_edges()
             for v in G.nodes()}

F_struct = np.array([[deg_dict[v], clust[v], avg_nbr_deg[v], ego_edges[v]]
                     for v in range(data_anom.num_nodes)])

iso = IsolationForest(contamination=float(y_true.mean()),
                      random_state=SEED, n_estimators=200)
iso.fit(F_struct)
# Higher decision_function => more normal; flip sign for anomaly score
baseline_scores = -iso.decision_function(F_struct)

report("DOMINANT (alpha=0.5)", y_true, scores)
report("IsolationForest base", y_true, baseline_scores)


alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
results = []
for a in alphas:
    m, _ = train_dominant(data_anom, hat_A, alpha=a, n_epochs=150, verbose=False)
    s, _, _ = score_dominant(m, X, hat_A, A_dense, alpha=a)
    auroc = roc_auc_score(y_true, s)
    auprc = average_precision_score(y_true, s)
    results.append((a, auroc, auprc))
    print(f"alpha={a:.1f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

res = np.array(results)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(res[:, 0], res[:, 1], marker="o", label="AUROC", color="#1565c0")
ax.plot(res[:, 0], res[:, 2], marker="s", label="AUPRC", color="#c62828")
ax.set_xlabel(r"$\alpha$  (attribute weight)")
ax.set_ylabel("score")
ax.set_title(r"DOMINANT performance vs. $\alpha$")
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()

