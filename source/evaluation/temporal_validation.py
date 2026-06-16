import torch
import time
import re
import numpy as np
from typing import List, Tuple, Any
from sklearn.metrics import f1_score, average_precision_score

from config import Config
from models.temporal_head import SnapshotEmbedder, TemporalLSTM, SnapshotEMA, LSTMConditionedHead
from evaluation.validation import _compute_class_weights, _find_best_f1_threshold, _aggregate_walk_forward
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


def train_lstm_conditioned(
    dm: Any,
    train_steps: List[int],
    cfg: Config,
    device: torch.device,
    epochs: int = 100,
    embed_dim: int = 32,
) -> Tuple[SnapshotEmbedder, TemporalLSTM, LSTMConditionedHead]:
    """
    Trains the end-to-end LSTM conditioned model on a sequence of snapshots.
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
    
    sorted_train_steps = sorted([t for t in train_steps if t in dm.graphs])
    if not sorted_train_steps:
        return embedder, lstm, head
        
    ytr_all = torch.cat([dm.graphs[t]["y"][dm.graphs[t]["labeled_mask"]] for t in sorted_train_steps])
    cls_w = _compute_class_weights(ytr_all, device)
    loss_fn = build_loss(cfg, cls_w)
    
    # GRAD FLOW GUARD: verify all modules receive gradients before real training
    embedder.train()
    lstm.train()
    head.train()
    
    t0 = sorted_train_steps[0]
    g0 = dm.graphs[t0]
    m0 = g0["labeled_mask"]
    # We need at least one labeled node for the smoke test
    if m0.sum() > 0:
        smoke_x = g0["prop"].to(device)
        smoke_y = g0["y"][m0].to(device)
        smoke_emb = embedder(smoke_x).unsqueeze(0).unsqueeze(0) # (1, 1, embed_dim)
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
    
    for epoch in range(epochs):
        opt.zero_grad()
        
        # 1. Compute embeddings for all historical snapshots in sequence
        embeddings = []
        for t in sorted_train_steps:
            g = dm.graphs[t]
            emb = embedder(g["prop"].to(device))
            embeddings.append(emb)
        embeddings_tensor = torch.stack(embeddings) # (T, embed_dim)
        
        # 2. Pass sequence through LSTM
        hidden_states = lstm(embeddings_tensor) # (T, hidden_dim)
        
        # 3. Compute loss over all snapshots
        total_loss = 0.0
        for i, t in enumerate(sorted_train_steps):
            g = dm.graphs[t]
            m = g["labeled_mask"]
            if m.sum() > 0:
                h_t = hidden_states[i]
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
    embed_dim: int = 32,
    epochs: int = 100,
) -> Tuple[float, float]:
    """Walk-forward evaluation of LSTM conditioned head."""
    safe_name = re.sub(r"[^\w\-]", "_", sweep_name)
    
    y_true_all, y_pred_all, s_pred_all = [], [], []
    
    for tau in cfg.test_steps:
        train_block, calib_step = _walk_forward_blocks(dm.graphs, tau)

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
        # Calibrate threshold on tau-1
        if calib_step in dm.graphs:
            cal_block = train_block + [calib_step]
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        # We need the LSTM state for calib_step.
                        # This requires running the embedder on train_block + calib_step
                        embeddings = []
                        for t in cal_block:
                            emb = embedder(dm.graphs[t]["prop"].to(device))
                            embeddings.append(emb)
                        embeddings_tensor = torch.stack(embeddings)
                        hidden_states = lstm(embeddings_tensor)
                        h_cal = hidden_states[-1]
                        
                        logits_cal = head(g_cal["prop"][m_cal].to(device), h_cal)
                        s_cal = torch.softmax(logits_cal, dim=1)[:, 1].cpu().numpy()
                    threshold = _find_best_f1_threshold(y_cal, s_cal)
                    
        # Test on tau
        with torch.no_grad():
            test_block = train_block + [calib_step, tau] if calib_step in dm.graphs else train_block + [tau]
            embeddings = []
            for t in test_block:
                if t in dm.graphs:
                    emb = embedder(dm.graphs[t]["prop"].to(device))
                    embeddings.append(emb)
            embeddings_tensor = torch.stack(embeddings)
            hidden_states = lstm(embeddings_tensor)
            h_tau = hidden_states[-1]
            
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
    embed_dim: int = 32,
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
    
    for epoch in range(epochs):
        opt.zero_grad()
        embedder.train()
        head.train()
        
        embeddings = []
        for t in sorted_train_steps:
            g = dm.graphs[t]
            emb = embedder(g["prop"].to(device))
            embeddings.append(emb)
        embeddings_tensor = torch.stack(embeddings)
        
        hidden_states = ema(embeddings_tensor)
        
        total_loss = 0.0
        for i, t in enumerate(sorted_train_steps):
            g = dm.graphs[t]
            m = g["labeled_mask"]
            if m.sum() > 0:
                h_t = hidden_states[i]
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
    embed_dim: int = 32,
    epochs: int = 100,
    alpha: float = 0.3,
) -> Tuple[float, float]:
    """Walk-forward evaluation of EMA conditioned head."""
    safe_name = re.sub(r"[^\w\-]", "_", sweep_name)
    
    y_true_all, y_pred_all, s_pred_all = [], [], []
    
    for tau in cfg.test_steps:
        train_block, calib_step = _walk_forward_blocks(dm.graphs, tau)

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
        head.eval()
        
        threshold = 0.5
        if calib_step in dm.graphs:
            cal_block = train_block + [calib_step]
            g_cal = dm.graphs[calib_step]
            m_cal = g_cal["labeled_mask"]
            if m_cal.sum() > 0:
                y_cal = g_cal["y"][m_cal].numpy()
                if len(np.unique(y_cal)) >= 2:
                    with torch.no_grad():
                        embeddings = []
                        for t in cal_block:
                            emb = embedder(dm.graphs[t]["prop"].to(device))
                            embeddings.append(emb)
                        embeddings_tensor = torch.stack(embeddings)
                        hidden_states = ema(embeddings_tensor)
                        h_cal = hidden_states[-1]
                        
                        logits_cal = head(g_cal["prop"][m_cal].to(device), h_cal)
                        s_cal = torch.softmax(logits_cal, dim=1)[:, 1].cpu().numpy()
                    threshold = _find_best_f1_threshold(y_cal, s_cal)
                    
        with torch.no_grad():
            test_block = train_block + [calib_step, tau] if calib_step in dm.graphs else train_block + [tau]
            embeddings = []
            for t in test_block:
                if t in dm.graphs:
                    emb = embedder(dm.graphs[t]["prop"].to(device))
                    embeddings.append(emb)
            embeddings_tensor = torch.stack(embeddings)
            hidden_states = ema(embeddings_tensor)
            h_tau = hidden_states[-1]
            
            logits_te = head(g["prop"][m].to(device), h_tau)
            s = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()
            
        y_pred = (s >= threshold).astype(int)
        y_true_all.append(yte_w)
        y_pred_all.append(y_pred)
        s_pred_all.append(s)
        
    pooled_f1, pooled_prauc, _, _, _ = _aggregate_walk_forward(y_true_all, y_pred_all, s_pred_all)
    return pooled_f1, pooled_prauc
