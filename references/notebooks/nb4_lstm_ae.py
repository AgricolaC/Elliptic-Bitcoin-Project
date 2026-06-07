import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

def make_time_series(n=3000, n_features=3, seed=SEED):
    """Generate a multivariate time series with structured anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Base signals: seasonal + trend + noise
    X = np.zeros((n, n_features))
    periods = [50, 70, 90]
    phases  = [0, np.pi/4, np.pi/2]
    amps    = [1.0, 0.8, 1.2]
    for i in range(n_features):
        X[:, i] = (amps[i] * np.sin(2*np.pi*t / periods[i] + phases[i])
                   + 0.1 * t / n           # slow upward trend
                   + rng.normal(0, 0.15, n))

    labels = np.zeros(n, dtype=int)

    # Anomaly 1: Point anomalies (large spikes)
    point_idx = rng.choice(range(200, n-200), size=15, replace=False)
    for idx in point_idx:
        channel = rng.integers(0, n_features)
        X[idx, channel] += rng.choice([-1, 1]) * rng.uniform(3.0, 5.0)
        labels[idx] = 1

    # Anomaly 2: Collective anomalies (window with mean shift)
    collective_starts = [700, 1400, 2100]
    for start in collective_starts:
        end = start + rng.integers(15, 30)
        shift = rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
        X[start:end, :] += shift
        labels[start:end] = 1

    # Anomaly 3: Contextual anomalies (wrong-phase oscillation)
    ctx_starts = [500, 1100, 1800, 2500]
    for start in ctx_starts:
        end = start + 20
        for i in range(n_features):
            # inject reversed-phase segment
            sub = np.arange(end-start)
            X[start:end, i] = amps[i] * np.sin(2*np.pi*sub/periods[i] + phases[i] + np.pi)
        labels[start:end] = 1

    return X.astype(np.float32), labels

X_raw, labels = make_time_series(n=3000, n_features=3)
print(f'Time series shape: {X_raw.shape}')
print(f'Anomaly rate: {labels.mean():.2%}  ({labels.sum()} anomalous timesteps)')

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
colors = ['steelblue', 'seagreen', 'darkorange']
names  = ['Sensor A', 'Sensor B', 'Sensor C']

anom_mask = labels == 1
for i, (ax, col, name) in enumerate(zip(axes, colors, names)):
    ax.plot(X_scaled[:, i], lw=0.7, color=col, alpha=0.8, label=name)
    # Shade anomaly regions
    in_anomaly = False
    start = 0
    for t in range(len(labels)+1):
        if t < len(labels) and labels[t] == 1 and not in_anomaly:
            start = t; in_anomaly = True
        elif (t == len(labels) or labels[t] == 0) and in_anomaly:
            ax.axvspan(start, t, alpha=0.25, color='tomato', label='Anomaly' if start==start else '')
            in_anomaly = False
    ax.set_ylabel(name, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Timestep')
fig.suptitle('Synthetic multivariate time series with labelled anomalies', fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/ts_overview.png', bbox_inches='tight')
plt.show()

def sliding_windows(X, labels, window_size=50, step=1):
    """Extract sliding windows; label = 1 if any anomaly in window."""
    windows, win_labels = [], []
    for i in range(0, len(X) - window_size + 1, step):
        windows.append(X[i:i+window_size])
        win_labels.append(int(labels[i:i+window_size].any()))
    return np.array(windows, dtype=np.float32), np.array(win_labels)

WINDOW = 50
# Use first 2000 steps for training (mostly normal), rest for test
train_cutoff = 2000
X_train_all, y_train_all = sliding_windows(X_scaled[:train_cutoff], labels[:train_cutoff], WINDOW)
X_test_w,    y_test_w    = sliding_windows(X_scaled,                 labels,                WINDOW)

# Keep only normal windows for training
X_train_w = X_train_all[y_train_all == 0]
print(f'Train windows (normal only): {X_train_w.shape}')
print(f'Test windows:                {X_test_w.shape}  |  anomaly rate: {y_test_w.mean():.2%}')

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h, c) = self.lstm(x)
        return out[:, -1, :]  # last hidden state as bottleneck


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, seq_len, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # Repeat bottleneck vector across time → feed as input to decoder LSTM
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq, hidden)
        out, _ = self.lstm(z_rep)
        return self.out(out)   # (batch, seq, features)


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size=64, seq_len=50, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, input_size, seq_len, num_layers, dropout)

    def forward(self, x):
        z = self.encoder(x)
        # Decode in reverse order (flip output, compare to flipped input)
        x_hat = self.decoder(z)
        return torch.flip(x_hat, dims=[1])   # reverse reconstruction

    def anomaly_score(self, x):
        """Per-window MSE score."""
        with torch.no_grad():
            x_hat = self(x)
            return ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()


def train_lstm_ae(model, X_windows, epochs=50, batch_size=64, lr=1e-3):
    model.to(DEVICE)
    t_data = torch.tensor(X_windows)
    loader = DataLoader(TensorDataset(t_data), batch_size=batch_size, shuffle=True)
    opt    = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            x_hat = model(xb)
            loss = nn.MSELoss()(x_hat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        losses.append(ep_loss / len(X_windows))
        if (ep + 1) % 10 == 0:
            print(f'  Epoch {ep+1:3d}/{epochs}  loss={losses[-1]:.5f}')
    return losses

lstm_ae = LSTMAE(input_size=3, hidden_size=64, seq_len=WINDOW)
print('Training LSTM-AE...')
lstm_losses = train_lstm_ae(lstm_ae, X_train_w, epochs=50)

lstm_ae.eval()
X_test_t = torch.tensor(X_test_w).to(DEVICE)
scores   = lstm_ae.anomaly_score(X_test_t)

auc  = roc_auc_score(y_test_w, scores)
ap   = average_precision_score(y_test_w, scores)
print(f'AUC-ROC: {auc:.4f}   Average Precision: {ap:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Score distribution
ax = axes[0]
ax.hist(scores[y_test_w==0], bins=80, alpha=0.6, density=True, color='steelblue', label='Normal')
ax.hist(scores[y_test_w==1], bins=80, alpha=0.6, density=True, color='tomato',   label='Anomaly')
tau = np.percentile(scores[y_test_w==0], 97)
ax.axvline(tau, lw=2, ls='--', color='darkorange', label=f'τ (Q97 normal)={tau:.3f}')
ax.set_xlabel('Reconstruction error'); ax.set_ylabel('Density')
ax.set_title('LSTM-AE score distributions', fontweight='bold')
ax.legend(fontsize=9)

# ROC curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test_w, scores)
ax.plot(fpr, tpr, lw=2, color='steelblue', label=f'LSTM-AE (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC curve', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/lstm_ae_scores.png', bbox_inches='tight')
plt.show()

# Compute per-timestep score by centering window on each timestep
def pointwise_scores(model, X, window_size, batch_size=256):
    """Assign each timestep the anomaly score of the window centred on it."""
    n = len(X)
    half = window_size // 2
    # Pad the series
    X_pad = np.pad(X, ((half, half), (0, 0)), mode='edge')
    windows = np.array([X_pad[i:i+window_size] for i in range(n)], dtype=np.float32)
    model.eval()
    all_scores = []
    for i in range(0, len(windows), batch_size):
        w = torch.tensor(windows[i:i+batch_size]).to(DEVICE)
        all_scores.append(model.anomaly_score(w))
    return np.concatenate(all_scores)

pt_scores = pointwise_scores(lstm_ae, X_scaled, WINDOW)

# Smooth for display
from scipy.ndimage import uniform_filter1d
pt_smooth = uniform_filter1d(pt_scores, size=15)

tau_pt = np.percentile(pt_smooth[:train_cutoff], 97)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                          gridspec_kw={'height_ratios': [2, 2, 2, 1.5]})

colors_ch = ['steelblue', 'seagreen', 'darkorange']
for i, (ax, col, name) in enumerate(zip(axes[:3], colors_ch, ['A', 'B', 'C'])):
    ax.plot(X_scaled[:, i], lw=0.6, color=col, alpha=0.85)
    ax.set_ylabel(f'Sensor {name}', fontsize=9)

# Score panel
ax = axes[3]
ax.plot(pt_smooth, lw=1, color='purple', alpha=0.9, label='Anomaly score (smoothed)')
ax.axhline(tau_pt, color='tomato', lw=1.5, ls='--', label=f'Threshold τ={tau_pt:.3f}')
ax.fill_between(range(len(pt_smooth)), 0, pt_smooth,
                where=pt_smooth > tau_pt, alpha=0.35, color='tomato', label='Detected')
ax.set_ylabel('Score', fontsize=9)
ax.set_xlabel('Timestep')
ax.legend(fontsize=8, loc='upper right')

# Shade true anomalies on all panels
for ax in axes:
    in_a = False; start_a = 0
    for t in range(len(labels)+1):
        if t < len(labels) and labels[t]==1 and not in_a:
            start_a=t; in_a=True
        elif (t==len(labels) or labels[t]==0) and in_a:
            ax.axvspan(start_a, t, alpha=0.15, color='red')
            in_a=False

axes[0].set_title('Multivariate time series + LSTM-AE anomaly score', fontweight='bold')

# Legend patch
red_patch = mpatches.Patch(color='red', alpha=0.3, label='True anomaly')
axes[0].legend(handles=[red_patch], loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('/tmp/temporal_score.png', bbox_inches='tight')
plt.show()

from scipy.stats import genpareto
from sklearn.metrics import f1_score

# Compute scores on training windows (all normal)
train_scores = lstm_ae.anomaly_score(torch.tensor(X_train_w).to(DEVICE))

# --- Strategy 1: Percentile ---
thresh_q95 = np.percentile(train_scores, 95)
thresh_q97 = np.percentile(train_scores, 97)
thresh_q99 = np.percentile(train_scores, 99)

# --- Strategy 2: Mean + k·std ---
mu, sigma = train_scores.mean(), train_scores.std()
thresh_3s = mu + 3 * sigma
thresh_4s = mu + 4 * sigma

# --- Strategy 3: EVT (fit GPD to tail) ---
tail_threshold = np.percentile(train_scores, 90)  # fit only on top 10%
tail_data = train_scores[train_scores > tail_threshold] - tail_threshold
try:
    c, loc, scale = genpareto.fit(tail_data, floc=0)
    # Threshold at 0.1% exceedance probability
    evt_thresh = tail_threshold + genpareto.ppf(0.999, c, loc=loc, scale=scale)
    print(f'EVT threshold (GPD, p=0.001): {evt_thresh:.4f}')
except Exception as e:
    evt_thresh = thresh_q99
    print(f'EVT fitting failed: {e}  → fallback to Q99')

# --- Strategy 4: Oracle F1 ---
cands = np.percentile(scores, np.arange(70, 100, 0.5))
best_f1, best_thresh_oracle = 0, 0
for c in cands:
    f1 = f1_score(y_test_w, (scores >= c).astype(int))
    if f1 > best_f1:
        best_f1 = f1; best_thresh_oracle = c

# Summary table
strategies = {
    'Q95':       thresh_q95,
    'Q97':       thresh_q97,
    'Q99':       thresh_q99,
    'μ + 3σ':   thresh_3s,
    'μ + 4σ':   thresh_4s,
    'EVT':       evt_thresh,
    'Oracle F1': best_thresh_oracle,
}
print('\nThreshold comparison:')
print(f'  {"Strategy":<15} {"Threshold":>10} {"F1":>8} {"Prec":>8} {"Recall":>8}')
from sklearn.metrics import precision_score, recall_score
for name, thresh in strategies.items():
    preds = (scores >= thresh).astype(int)
    f1_s   = f1_score(y_test_w, preds, zero_division=0)
    prec_s = precision_score(y_test_w, preds, zero_division=0)
    rec_s  = recall_score(y_test_w, preds, zero_division=0)
    print(f'  {name:<15} {thresh:>10.4f} {f1_s:>8.3f} {prec_s:>8.3f} {rec_s:>8.3f}')

# Visualise threshold positions on score distribution
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(train_scores, bins=80, density=True, alpha=0.5, color='steelblue', label='Train scores (normal)')

cols = ['#2196F3','#4CAF50','#9C27B0','darkorange','#E91E63','tomato','black']
for (name, thresh), col in zip(strategies.items(), cols):
    ax.axvline(thresh, lw=2 if name=='Oracle F1' else 1.5,
               ls='-' if name=='Oracle F1' else '--', color=col, label=name)

ax.set_xlabel('Anomaly score'); ax.set_ylabel('Density')
ax.set_title('Threshold strategies on training score distribution', fontweight='bold')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('/tmp/threshold_strategies.png', bbox_inches='tight')
plt.show()

lstm_ae.eval()
with torch.no_grad():
    X_hat_t = lstm_ae(X_test_t)   # (N_windows, seq_len, features)

# Per-feature score: average over timesteps within window
per_feat_scores = ((X_test_t - X_hat_t) ** 2).mean(dim=1).cpu().numpy()  # (N, 3)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
feature_names = ['Sensor A', 'Sensor B', 'Sensor C']
colors_ch = ['steelblue', 'seagreen', 'darkorange']
for i, (ax, name, col) in enumerate(zip(axes, feature_names, colors_ch)):
    ax.hist(per_feat_scores[y_test_w==0, i], bins=60, alpha=0.6, density=True, color=col, label='Normal')
    ax.hist(per_feat_scores[y_test_w==1, i], bins=60, alpha=0.6, density=True, color='tomato', label='Anomaly')
    auc_f = roc_auc_score(y_test_w, per_feat_scores[:, i])
    ax.set_title(f'{name}\n(AUC={auc_f:.3f})', fontweight='bold')
    ax.set_xlabel('Recon. error')
    if i == 0:
        ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle('Per-feature reconstruction error distribution', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/per_feature_scores.png', bbox_inches='tight')
plt.show()

window_sizes = [20, 30, 50, 70, 100]
results_w = {}
print('Sweeping window sizes...')
for w in window_sizes:
    X_tr_w, y_tr_w = sliding_windows(X_scaled[:train_cutoff], labels[:train_cutoff], w)
    X_te_w, y_te_w = sliding_windows(X_scaled, labels, w)
    X_tr_w = X_tr_w[y_tr_w == 0]
    if len(X_tr_w) < 100:
        continue
    m = LSTMAE(input_size=3, hidden_size=64, seq_len=w)
    train_lstm_ae(m, X_tr_w, epochs=40, batch_size=64)
    m.eval()
    sc = m.anomaly_score(torch.tensor(X_te_w).to(DEVICE))
    auc_w = roc_auc_score(y_te_w, sc)
    ap_w  = average_precision_score(y_te_w, sc)
    results_w[w] = (auc_w, ap_w)
    print(f'  Window={w:4d}  AUC={auc_w:.4f}  AP={ap_w:.4f}')

fig, ax = plt.subplots(figsize=(7, 4))
ws  = list(results_w.keys())
aucs = [v[0] for v in results_w.values()]
aps  = [v[1] for v in results_w.values()]
ax.plot(ws, aucs, 'o-', color='steelblue', lw=2, ms=8, label='AUC-ROC')
ax.plot(ws, aps,  's--', color='tomato',    lw=2, ms=8, label='Avg Precision')
ax.set_xlabel('Window size $T$'); ax.set_ylabel('Score')
ax.set_title('LSTM-AE performance vs window size', fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/window_sweep.png', bbox_inches='tight')
plt.show()
