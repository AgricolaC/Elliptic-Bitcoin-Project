import torch
import time
import numpy as np
from typing import List, Tuple, Any
from sklearn.metrics import f1_score, average_precision_score

from config import Config
from models.temporal_head import SnapshotEmbedder, TemporalLSTM, SnapshotEMA, LSTMConditionedHead
from evaluation.validation import _compute_class_weights, _find_best_f1_threshold, _aggregate_walk_forward, _calibrate_threshold


def _train_illicit_rate(dm, cfg):
    """Global illicit base rate over the training window (for the ε-fallback)."""
    ys = [dm.graphs[t]["y"][dm.graphs[t]["labeled_mask"]].numpy()
          for t in cfg.train_steps if t in dm.graphs]
    if not ys:
        return 0.5
    y = np.concatenate(ys)
    return float((y == 1).mean()) if len(y) else 0.5
from models.classifier import build_loss


def _walk_forward_blocks(available_steps, tau):
    """Compute the training/calibration split for walk-forward step ``tau``.

    Returns ``(train_block, calib_step)`` where ``calib_step == tau - 1`` and
    every step in ``train_block`` is strictly before ``tau - 1`` — i.e. training
    never sees the calibration step or τ itself. Steps not present in
    ``available_steps`` are skipped. Shared by both the LSTM and EMA evaluators
    so their windows cannot drift apart.
    """
    available = set(available_steps)
    calib_step = tau - 1
    train_block = [t for t in range(min(available_steps), tau - 1) if t in available]
    assert all(t < tau - 1 for t in train_block), \
        f"train_block leaks calib/test step: max={max(train_block)} tau={tau}"
    return train_block, calib_step


def _onestep_blocks(available_steps, tau):
    """One-step-ahead walk-forward blocks for step ``tau``.

    Returns ``(train_block, calib_step, calib_state_steps, infer_state_steps)``.

    DESIGN — One-step-ahead (P0a). The hidden state that classifies a step is
    built ONLY from steps strictly before it, so the model never sees the
    snapshot it is scoring (removes the self-conditioning confound, review #2):
      - ``infer_state_steps`` classifies τ   = train_block + [calib_step]  (excludes τ)
      - ``calib_state_steps`` classifies τ-1 = train_block                 (excludes τ-1)
    Shared by the LSTM and EMA evaluators so their windows cannot drift apart.
    """
    train_block, calib_step = _walk_forward_blocks(available_steps, tau)
    calib_state_steps = list(train_block)
    infer_state_steps = train_block + [calib_step]
    assert tau not in infer_state_steps, f"τ={tau} leaked into the state that classifies τ"
    assert calib_step not in calib_state_steps, \
        f"calib step {calib_step} leaked into the state that classifies τ-1"
    return train_block, calib_step, calib_state_steps, infer_state_steps


def _temporal_state(embedder, temporal, steps, dm, device):
    """Run the (label-free) embedder over ``steps`` in order, pass the sequence
    through ``temporal`` (TemporalLSTM or SnapshotEMA — identical
    ``(T, D) -> (T, H)`` interface), and return the hidden state after the last
    step. Steps absent from ``dm.graphs`` are skipped.

    One-step-ahead: callers pass the history EXCLUDING the step being classified
    (see ``_onestep_blocks``), so the returned state never incorporates the
    scored snapshot's own embedding.

    h_0 reset guarantee: torch.stack() builds a fresh sequence each call;
    TemporalLSTM and SnapshotEMA both start from a zero initial state. There
    is no persistent hidden state across calls.
    """
    embeddings = [embedder(dm.graphs[t]["prop"].to(device)) for t in steps if t in dm.graphs]
    assert len(embeddings) > 0, (
        f"_temporal_state: no valid steps in {steps} — "
        f"dm.graphs has {len(dm.graphs)} entries: {sorted(dm.graphs)[:5]}..."
    )
    hidden_states = temporal(torch.stack(embeddings))
    return hidden_states[-1]


def train_lstm_conditioned(
    dm: Any,
    train_steps: List[int],
    cfg: Config,
    device: torch.device,
    epochs: int = 100,
    embed_dim: int = 128,
    shuffle_train: bool = False,
) -> Tuple[SnapshotEmbedder, TemporalLSTM, LSTMConditionedHead]:
    """
    Trains the end-to-end LSTM conditioned model on a sequence of snapshots.

    Returns (embedder, lstm, head) all in train mode. Callers that run inference
    immediately after must call embedder.eval(); lstm.eval(); head.eval() first.
    """
    embedder = SnapshotEmbedder(dm.sgc_input_dim, embed_dim, cfg).to(device)
    lstm = TemporalLSTM(embed_dim, cfg.lstm_hidden).to(device)
    head = LSTMConditionedHead(dm.sgc_input_dim, cfg.lstm_hidden, cfg).to(device)
    
    all_params = (
        list(embedder.parameters()) +
        list(lstm.parameters()) +
        list(head.parameters())
    )
    opt = torch.optim.AdamW(all_params, lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay)
    
    base_train_steps = sorted([t for t in train_steps if t in dm.graphs])
    if not base_train_steps:
        return embedder, lstm, head
        
    ytr_all = torch.cat([dm.graphs[t]["y"][dm.graphs[t]["labeled_mask"]] for t in base_train_steps])
    cls_w = _compute_class_weights(ytr_all, device)
    loss_fn = build_loss(cfg, cls_w)
    
    # GRAD FLOW GUARD: verify all modules receive gradients before real training
    embedder.train()
    lstm.train()
    head.train()
    
    t0 = base_train_steps[0]
    g0 = dm.graphs[t0]
    m0 = g0["labeled_mask"]
    # We need at least one labeled node for the smoke test
    if m0.sum() > 0:
        smoke_x = g0["prop"].to(device)
        smoke_y = g0["y"][m0].to(device)
        smoke_emb_vec = embedder(smoke_x)
        assert smoke_emb_vec.shape == (embed_dim,), \
            f"SMOKE: SnapshotEmbedder output shape {smoke_emb_vec.shape} != ({embed_dim},)"
        smoke_emb = smoke_emb_vec.unsqueeze(0).unsqueeze(0) # (1, 1, embed_dim)
        smoke_lstm_out, _ = lstm.lstm(smoke_emb)
        smoke_h = smoke_lstm_out.squeeze(0).squeeze(0) # (hidden_dim,)
        smoke_logits = head(smoke_x[m0], smoke_h)
        smoke_loss = loss_fn(smoke_logits, smoke_y)
        smoke_loss.backward()
        
        for name, param in [("embedder", embedder), ("lstm", lstm), ("head", head)]:
            for pname, p in param.named_parameters():
                assert p.grad is not None, \
                    f"GRAD FLOW GUARD: {name}.{pname} received no gradient — optimizer would skip it"
        opt.zero_grad()
    
    import random
    for epoch in range(epochs):
        opt.zero_grad()
        
        # Shuffle order per epoch if control test is active
        seq_steps = base_train_steps.copy()
        if shuffle_train:
            random.shuffle(seq_steps)
            
        # 1. Compute embeddings for all historical snapshots in sequence
        embeddings = []
        for t in seq_steps:
            g = dm.graphs[t]
            emb = embedder(g["prop"].to(device))
            embeddings.append(emb)
        embeddings_tensor = torch.stack(embeddings) # (T, embed_dim)
        
        # 2. Pass sequence through LSTM
        hidden_states = lstm(embeddings_tensor) # (T, hidden_dim)
        
        # CAUSAL SHIFT: h_t must only see steps [0 ... t-1] to prevent leakage
        shifted_hidden_states = torch.zeros_like(hidden_states)
        if hidden_states.size(0) > 1:
            shifted_hidden_states[1:] = hidden_states[:-1]
        
        # 3. Compute loss over all snapshots
        total_loss = 0.0
        for i, t in enumerate(seq_steps):
            g = dm.graphs[t]
            m = g["labeled_mask"]
            if m.sum() > 0:
                h_t = shifted_hidden_states[i]
                logits = head(g["prop"][m].to(device), h_t)
                total_loss += loss_fn(logits, g["y"][m].to(device))

        if total_loss > 0:
            total_loss.backward()
            opt.step()

    return embedder, lstm, head


def walk_forward_lstm_conditioned(
    dm: Any,
    cfg: Config,
    device: torch.device,
    sweep_name: str = "LSTM",
    embed_dim: int = 128,
    epochs: int = 100,
) -> Tuple[float, float]:
    """Walk-forward evaluation of LSTM conditioned head."""
    y_true_all, y_pred_all, s_pred_all = [], [], []
    global_illicit_rate = _train_illicit_rate(dm, cfg)

    for tau in cfg.test_steps:
        train_block, calib_step, calib_state, infer_state = _onestep_blocks(dm.graphs, tau)

        if not train_block: continue

        g = dm.graphs[tau]
        m = g["labeled_mask"]
        if m.sum() == 0: continue
        yte_w = g["y"][m].numpy()
        if len(np.unique(yte_w)) < 2: continue

        embedder, lstm, head = train_lstm_conditioned(
            dm, train_block, cfg, device, epochs=epochs, embed_dim=embed_dim
        )

        embedder.eval()
        lstm.eval()
        head.eval()

        threshold = 0.5
        # Calibrate threshold on tau-1 (one-step-ahead: state excludes tau-1)
        if calib_step in dm.graphs:
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        h_cal = _temporal_state(embedder, lstm, calib_state, dm, device)
                        logits_cal = head(g_cal["prop"][m_cal].to(device), h_cal)
                        s_cal = torch.softmax(logits_cal, dim=1)[:, 1].cpu().numpy()
                    threshold, _fallback_fired = _calibrate_threshold(
                        y_cal, s_cal, global_illicit_rate)

        # Test on tau (one-step-ahead: state excludes tau — see _onestep_blocks)
        with torch.no_grad():
            h_tau = _temporal_state(embedder, lstm, infer_state, dm, device)
            logits_te = head(g["prop"][m].to(device), h_tau)
            s = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()

        y_pred = (s >= threshold).astype(int)
        y_true_all.append(yte_w)
        y_pred_all.append(y_pred)
        s_pred_all.append(s)
        
    pooled_f1, pooled_prauc, _, _, _ = _aggregate_walk_forward(y_true_all, y_pred_all, s_pred_all)
    return pooled_f1, pooled_prauc


def train_ema_conditioned(
    dm: Any,
    train_steps: List[int],
    cfg: Config,
    device: torch.device,
    epochs: int = 100,
    embed_dim: int = 128,
    alpha: float = 0.3,
) -> Tuple[SnapshotEmbedder, SnapshotEMA, LSTMConditionedHead]:
    """
    Trains the EMA conditioned model on a sequence of snapshots.
    """
    embedder = SnapshotEmbedder(dm.sgc_input_dim, embed_dim, cfg).to(device)
    ema = SnapshotEMA(alpha=alpha).to(device)
    # The EMA output dimension is embed_dim
    head = LSTMConditionedHead(dm.sgc_input_dim, embed_dim, cfg).to(device)
    
    all_params = (
        list(embedder.parameters()) +
        list(head.parameters())
    )
    opt = torch.optim.AdamW(all_params, lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay)
    
    sorted_train_steps = sorted([t for t in train_steps if t in dm.graphs])
    if not sorted_train_steps:
        return embedder, ema, head
        
    ytr_all = torch.cat([dm.graphs[t]["y"][dm.graphs[t]["labeled_mask"]] for t in sorted_train_steps])
    cls_w = _compute_class_weights(ytr_all, device)
    loss_fn = build_loss(cfg, cls_w)
    
    embedder.train()
    head.train()

    for epoch in range(epochs):
        opt.zero_grad()

        embeddings = []
        for t in sorted_train_steps:
            g = dm.graphs[t]
            emb = embedder(g["prop"].to(device))
            embeddings.append(emb)
        embeddings_tensor = torch.stack(embeddings)
        
        hidden_states = ema(embeddings_tensor)
        
        # CAUSAL SHIFT: h_t must only see steps [0 ... t-1] to prevent leakage
        shifted_hidden_states = torch.zeros_like(hidden_states)
        if hidden_states.size(0) > 1:
            shifted_hidden_states[1:] = hidden_states[:-1]
        
        total_loss = 0.0
        for i, t in enumerate(sorted_train_steps):
            g = dm.graphs[t]
            m = g["labeled_mask"]
            if m.sum() > 0:
                h_t = shifted_hidden_states[i]
                logits = head(g["prop"][m].to(device), h_t)
                total_loss += loss_fn(logits, g["y"][m].to(device))
                
        if total_loss > 0:
            total_loss.backward()
            opt.step()
            
    return embedder, ema, head


def walk_forward_ema_conditioned(
    dm: Any,
    cfg: Config,
    device: torch.device,
    sweep_name: str = "EMA",
    embed_dim: int = 128,
    epochs: int = 100,
    alpha: float = 0.3,
) -> Tuple[float, float]:
    """Walk-forward evaluation of EMA conditioned head."""
    y_true_all, y_pred_all, s_pred_all = [], [], []
    global_illicit_rate = _train_illicit_rate(dm, cfg)

    for tau in cfg.test_steps:
        train_block, calib_step, calib_state, infer_state = _onestep_blocks(dm.graphs, tau)

        if not train_block: continue

        g = dm.graphs[tau]
        m = g["labeled_mask"]
        if m.sum() == 0: continue
        yte_w = g["y"][m].numpy()
        if len(np.unique(yte_w)) < 2: continue

        embedder, ema, head = train_ema_conditioned(
            dm, train_block, cfg, device, epochs=epochs, embed_dim=embed_dim, alpha=alpha
        )

        embedder.eval()
        ema.eval()
        head.eval()

        threshold = 0.5
        # Calibrate threshold on tau-1 (one-step-ahead: state excludes tau-1)
        if calib_step in dm.graphs:
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        h_cal = _temporal_state(embedder, ema, calib_state, dm, device)
                        logits_cal = head(g_cal["prop"][m_cal].to(device), h_cal)
                        s_cal = torch.softmax(logits_cal, dim=1)[:, 1].cpu().numpy()
                    threshold, _fallback_fired = _calibrate_threshold(
                        y_cal, s_cal, global_illicit_rate)

        # Test on tau (one-step-ahead: state excludes tau — see _onestep_blocks)
        with torch.no_grad():
            h_tau = _temporal_state(embedder, ema, infer_state, dm, device)
            logits_te = head(g["prop"][m].to(device), h_tau)
            s = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()

        y_pred = (s >= threshold).astype(int)
        y_true_all.append(yte_w)
        y_pred_all.append(y_pred)
        s_pred_all.append(s)
        
    pooled_f1, pooled_prauc, _, _, _ = _aggregate_walk_forward(y_true_all, y_pred_all, s_pred_all)
    return pooled_f1, pooled_prauc
