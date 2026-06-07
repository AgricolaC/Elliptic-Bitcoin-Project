import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
import umap

np.random.seed(0)
plt.rcParams['figure.dpi'] = 110

# ----------------- the four estimators -----------------

def id_pca(X, var_threshold=0.95):
    """PCA estimator: smallest d capturing var_threshold of variance."""
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    lam = s**2
    cum = np.cumsum(lam) / lam.sum()
    return int(np.searchsorted(cum, var_threshold) + 1)

def id_correlation(X, n_radii=40):
    """Grassberger-Procaccia: fit slope of log C(r) vs log r in the linear regime."""
    D = pdist(X)
    rs = np.exp(np.linspace(np.log(D.min() * 1.1), np.log(D.max() * 0.5), n_radii))
    Cs = np.array([(D < r).sum() * 2.0 / (len(X) * (len(X) - 1)) for r in rs])
    mask = (Cs > 1e-4) & (Cs < 0.5)        # the scaling regime
    if mask.sum() < 3:
        return np.nan
    slope = np.polyfit(np.log(rs[mask]), np.log(Cs[mask]), 1)[0]
    return slope

def id_levina_bickel(X, k=10):
    """Levina-Bickel MLE with MacKay-Ghahramani harmonic average."""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    d_nn, _ = nbrs.kneighbors(X)              # d_nn[:, 0] = 0 (self)
    T = d_nn[:, 1:]                            # T[:, j] = distance to (j+1)-th NN
    # local MLE for each point
    log_ratio = np.log(T[:, k-1:k] / T[:, :k-1])
    d_local = (k - 1) / log_ratio.sum(axis=1)
    # harmonic (inverse) mean
    return 1.0 / np.mean(1.0 / d_local)

def id_twonn(X, frac=0.9):
    """Two-NN estimator (Facco et al. 2017). Trim top (1-frac) for robustness."""
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    d_nn, _ = nbrs.kneighbors(X)
    r1, r2 = d_nn[:, 1], d_nn[:, 2]
    mu = r2 / r1
    mu = np.sort(mu)
    # use the linear part of the CDF in log-log
    cdf = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    keep = cdf < frac
    x = np.log(mu[keep])
    y = -np.log(1.0 - cdf[keep])
    slope, _ = np.polyfit(x, y, 1)
    return slope

def all_estimators(X, label):
    print(f"\n{label} (shape {X.shape})")
    print(f"  PCA (95% var):       d = {id_pca(X):>5}")
    print(f"  Correlation dim:     d = {id_correlation(X):>5.2f}")
    print(f"  Levina-Bickel (k=10):d = {id_levina_bickel(X, 10):>5.2f}")
    print(f"  TwoNN:               d = {id_twonn(X):>5.2f}")

# Three manifolds with known intrinsic dimension

# 1. Flat 5-D Gaussian inside R^20 (true d=5)
n = 3000
X_flat = np.zeros((n, 20))
X_flat[:, :5] = np.random.randn(n, 5)

# 2. Swiss roll in R^3 (true d=2)
X_swiss, _ = make_swiss_roll(n, noise=0.0, random_state=0)

# 3. Unit sphere S^2 in R^3 (true d=2)
X_sphere = np.random.randn(n, 3)
X_sphere /= np.linalg.norm(X_sphere, axis=1, keepdims=True)

all_estimators(X_flat,   "5-D Gaussian in R^20 (true d=5)")
all_estimators(X_swiss,  "Swiss roll       (true d=2)")
all_estimators(X_sphere, "Sphere S^2       (true d=2)")

# Sphere S^2 with full-dimensional noise
n = 5000
X = np.random.randn(n, 3)
X /= np.linalg.norm(X, axis=1, keepdims=True)
noise = 0.05 * np.random.randn(n, 3)
X_noisy = X + noise

# Compute Levina-Bickel dimension at a range of k values
ks = [2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500]
d_traj = [id_levina_bickel(X_noisy, k=k) for k in ks]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ks, d_traj, 'o-', color='#1C7293', lw=2)
ax.axhline(2.0, color='green', ls='--', label='true intrinsic d = 2')
ax.axhline(3.0, color='red', ls=':', label='ambient D = 3')
ax.set_xscale('log')
ax.set_xlabel('k (nearest neighbours used)')
ax.set_ylabel(r'Levina-Bickel $\hat{d}_k$')
ax.set_title('Dimension trajectory on a noisy sphere\n(sub-noise -> manifold plateau -> curvature inflation)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

digits = load_digits()
X_mnist = digits.data            # (1797, 64)
y_mnist = digits.target

print(f'MNIST shape: {X_mnist.shape}  (ambient D = {X_mnist.shape[1]})')
print()
all_estimators(X_mnist, 'MNIST (8x8 digits)')

per_class_d = []
for c in range(10):
    Xc = X_mnist[y_mnist == c]
    d = id_twonn(Xc)
    per_class_d.append(d)
    print(f'  digit {c}: n={len(Xc):>4}, TwoNN d = {d:.2f}')

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(10), per_class_d, color='#1C7293', alpha=0.85)
ax.axhline(np.mean(per_class_d), color='red', ls='--', label=f'mean = {np.mean(per_class_d):.2f}')
ax.set_xlabel('digit class')
ax.set_ylabel('TwoNN intrinsic dimension')
ax.set_title('MNIST: each digit class lies on a ~10-D submanifold of R^64')
ax.set_xticks(range(10)); ax.legend()
plt.tight_layout(); plt.show()

n_per_cluster = 1000

# Three Gaussian clusters in R^10
A = np.random.randn(n_per_cluster, 10) * 1.0                       # spread
B = np.random.randn(n_per_cluster, 10) * 0.1                       # TIGHT
C = np.random.randn(n_per_cluster, 10) * 1.0

# Place centres along axis 0, equal separations
B[:, 0] += 20
C[:, 0] += 40

X = np.vstack([A, B, C])
y = np.array([0]*n_per_cluster + [1]*n_per_cluster + [2]*n_per_cluster)
names = ['A (spread)', 'B (TIGHT)', 'C (spread)']

# Compute true cluster diameters and centre distances (ground truth)
print('True 10-D geometry:')
for k, lbl in enumerate(names):
    Xc = X[y == k]
    diam = np.std(Xc, axis=0).mean() * np.sqrt(10)
    print(f'  {lbl:>14}: radius ~ {diam:.2f}')
print(f'  centre A->B = 20.0, B->C = 20.0  (equal separations)')

tsne_emb = TSNE(n_components=2, perplexity=30, random_state=0,
                init='pca').fit_transform(X)
umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)

def measure_layout(emb, y):
    """Compute visual cluster radius (std-based) and centre-centre distance."""
    radii, centres = [], []
    for k in range(3):
        Z = emb[y == k]
        c = Z.mean(axis=0)
        centres.append(c)
        radii.append(np.linalg.norm(Z - c, axis=1).mean())
    centres = np.array(centres)
    d_AB = np.linalg.norm(centres[0] - centres[1])
    d_BC = np.linalg.norm(centres[1] - centres[2])
    return radii, d_AB, d_BC

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
colors = ['#1C7293', '#E76F51', '#147830']
for ax, emb, ttl in zip(axes, [tsne_emb, umap_emb], ['t-SNE', 'UMAP']):
    for k in range(3):
        ax.scatter(emb[y == k, 0], emb[y == k, 1],
                   c=colors[k], s=10, alpha=0.7, label=names[k])
    radii, d_AB, d_BC = measure_layout(emb, y)
    ax.set_title(f'{ttl}\nradii A,B,C = {radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f}'
                 f'  |  d(A,B)={d_AB:.2f}, d(B,C)={d_BC:.2f}')
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('The mass-of-clusters illusion: visual sizes/distances vs ground truth (B is 10x tighter; separations are equal)', fontsize=11)
plt.tight_layout(); plt.show()

# Sub-sample MNIST for speed
rng = np.random.RandomState(0)
idx = rng.choice(len(X_mnist), 800, replace=False)
Xs, ys = X_mnist[idx], y_mnist[idx]

perplexities = [5, 30, 100, 200]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, p in zip(axes, perplexities):
    emb = TSNE(n_components=2, perplexity=p, random_state=0, init='pca',
               max_iter=1000).fit_transform(Xs)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=ys, cmap='tab10', s=12)
    ax.set_title(f'perplexity = {p}')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('t-SNE on MNIST as perplexity varies: too-low creates spurious small clusters; too-high merges true ones',
             fontsize=11)
plt.tight_layout(); plt.show()

# Effect of min_dist (cluster compactness)
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, md in zip(axes, [0.0, 0.1, 0.5, 0.99]):
    emb = umap.UMAP(n_neighbors=15, min_dist=md, random_state=0).fit_transform(Xs)
    ax.scatter(emb[:, 0], emb[:, 1], c=ys, cmap='tab10', s=12)
    ax.set_title(f'min_dist = {md}')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('UMAP min_dist sweep: smaller = tighter clusters; larger = uniform spread', fontsize=11)
plt.tight_layout(); plt.show()

# Effect of n_neighbors (locality scale)
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, k in zip(axes, [5, 15, 50, 200]):
    emb = umap.UMAP(n_neighbors=k, min_dist=0.1, random_state=0).fit_transform(Xs)
    ax.scatter(emb[:, 0], emb[:, 1], c=ys, cmap='tab10', s=12)
    ax.set_title(f'n_neighbors = {k}')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('UMAP n_neighbors sweep: smaller = more local clusters; larger = more global structure',
             fontsize=11)
plt.tight_layout(); plt.show()

from sklearn.manifold import trustworthiness

# Step 1: measure intrinsic dim
d_hat = id_twonn(X_mnist)
print(f'STEP 1 — Estimated intrinsic dim of MNIST: d_hat = {d_hat:.2f}')
print(f'         (vs ambient D = {X_mnist.shape[1]})')

# Step 2: choose embedding dim (2D for vis; 2*d for downstream pipelines)
print(f'\nSTEP 2 — For visualisation we use d=2 (with expected distortion).')
print(f'         For a downstream classifier we would use d ~= 2*d_hat = {2*int(d_hat)}.')

# Step 3: compute embeddings
print('\nSTEP 3 — Computing embeddings...')
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=0, init='pca').fit_transform(X_mnist)
umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X_mnist)
umap_hi = umap.UMAP(n_neighbors=15, n_components=int(2*d_hat),
                    random_state=0).fit_transform(X_mnist)
print(f'   tsne (2D):       {tsne_2d.shape}')
print(f'   umap (2D):       {umap_2d.shape}')
print(f'   umap ({2*int(d_hat)}D pipeline): {umap_hi.shape}')

# Step 4: validate quality via trustworthiness
# (How well do low-dim neighbours match high-dim neighbours?)
print('\nSTEP 4 — Trustworthiness (closer to 1 = better local neighbourhood preservation)')
for name, emb in [('t-SNE 2D', tsne_2d), ('UMAP 2D', umap_2d),
                  (f'UMAP {2*int(d_hat)}D', umap_hi)]:
    t = trustworthiness(X_mnist, emb, n_neighbors=10)
    print(f'   {name:<20}: T(k=10) = {t:.3f}')

# Visualise the two 2D embeddings side by side, colour by class
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, emb, ttl in zip(axes, [tsne_2d, umap_2d], ['t-SNE', 'UMAP']):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_mnist, cmap='tab10', s=14)
    ax.set_title(ttl)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, ticks=range(10))
plt.suptitle('Pipeline output: MNIST embedded after intrinsic-dimension analysis')
plt.tight_layout(); plt.show()
