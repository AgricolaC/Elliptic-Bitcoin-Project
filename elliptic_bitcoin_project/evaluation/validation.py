import numpy as np
import torch
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Any
from sklearn.metrics import f1_score, average_precision_score
from config import Config, OUTPUT_DIR, set_global_seeds

try:
    from models.classifier import SGCHead, build_loss
except ImportError:
    SGCHead = build_loss = None

def _aggregate_walk_forward(y_true_list: List[np.ndarray], y_pred_list: List[np.ndarray], score_list: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Compute pooled and macro-averaged walk-forward metrics.
    Returns: (pooled_f1, pooled_prauc, macro_f1, macro_prauc)
    """
    yt_p = np.concatenate(y_true_list)
    yp_p = np.concatenate(y_pred_list)
    ys_p = np.concatenate(score_list)
    
    pooled_f1 = float(f1_score(yt_p, yp_p, pos_label=1, zero_division=0))
    pooled_prauc = float(average_precision_score(yt_p, ys_p))
    
    f1s, praucs = [], []
    for yt, yp, ys in zip(y_true_list, y_pred_list, score_list):
        if len(np.unique(yt)) >= 2:
            f1s.append(float(f1_score(yt, yp, pos_label=1, zero_division=0)))
            praucs.append(float(average_precision_score(yt, ys)))
        else:
            f1s.append(0.0)
            praucs.append(0.0)
            
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_prauc = float(np.mean(praucs)) if praucs else 0.0
    
    return pooled_f1, pooled_prauc, macro_f1, macro_prauc


def stack_prop(dm: Any, steps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ALL nodes' propagated features over `steps`."""
    Xs, ys = [], []
    for t in steps:
        g = dm.graphs[t]
        Xs.append(g["prop"])
        ys.append(g["y"])
    return torch.cat(Xs), torch.cat(ys)


def fit_head(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    in_dim: int,
    cfg: Config,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = -1,
) -> "SGCHead":
    """
    Train one SGCHead.

    Defensive Notes:
        - Seeds reset before every fit call for reproducibility.
        - Labeled mask applied inside loop: unknown nodes (-1) excluded from loss.
    """
    set_global_seeds(cfg.seed)
    model = SGCHead(in_dim, cfg).to(device)
    loss_fn = build_loss(cfg, class_weights)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.sgc_lr, weight_decay=cfg.sgc_weight_decay
    )
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    mask = (ytr != -1)

    model.train()
    n_epochs = epochs if epochs > 0 else cfg.sgc_epochs
    for _ in range(n_epochs):
        opt.zero_grad()
        if mask.sum() > 0:
            loss = loss_fn(model(Xtr)[mask], ytr[mask])
            loss.backward()
            opt.step()
    return model


def _compute_class_weights(ytr: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from labeled nodes in ytr.

    Defensive Notes:
        - Only labeled nodes (y != -1) are used for counting.
        - Returns uniform weights if no labeled nodes exist.
    """
    labeled = ytr[ytr != -1]
    if labeled.numel() == 0:
        return torch.ones(2, device=device)
    counts = torch.bincount(labeled, minlength=2).float()
    # Guard against a class being absent in this window
    counts = counts.clamp(min=1.0)
    return (counts.sum() / (2.0 * counts)).to(device)


def walk_forward_validation(
    dm: Any,
    cfg: Config,
    device: torch.device,
    sweep_name: str = "default",   # W7 FIX: parameterized filename
    return_records: bool = False,
    window: int = None,
) -> Any:
    """
    Walk-forward temporal validation with dynamic class weights.

    W7 FIX: Output filename embeds sweep_name, preventing silent overwrite when
            multiple sweeps are run sequentially.

    W8 FIX: Class weights are recomputed inside the tau loop from the expanding
            training window [1..tau-1], not from a fixed static training split.
            This corrects miscalibration as the class distribution shifts over time.

    Defensive Notes:
        - LEAKAGE GUARD: train_block = [1, tau-1]; assert max == tau-1 each step.
        - Skips tau steps with no labeled nodes or single-class labels.
        - Class weights are computed before each model fit from actual window data.

    Args:
        dm:             EllipticDataModule with 'prop' keys set (by setup()).
        cfg:            Configuration object.
        device:         Torch device for model training.
        sweep_name:     Label embedded in the output filename.
        return_records: If True, return per-step records as third element.
        window:         If set, use a sliding window of this width. None = expanding.

    Returns:
        If return_records:  (pooled_f1, pooled_prauc, wf_records)
        Else:               (pooled_f1, pooled_prauc)
    """
    # Sanitize sweep_name for safe filesystem use
    safe_name = re.sub(r"[^\w\-]", "_", sweep_name)

    wf_steps = []
    wf_f1_per_step = []
    wf_prauc_per_step = []
    wf_records = []
    y_true_all, y_pred_all, s_pred_all = [], [], []
    total_wf_train_time = 0.0

    for tau in cfg.test_steps:
        start_t = max(min(dm.graphs), tau - window) if window else min(dm.graphs)
        train_block = list(range(start_t, tau))
        train_block = [t for t in train_block if t in dm.graphs]

        if not train_block:
            continue

        # LEAKAGE GUARD: no test-step data can be in the training window
        assert max(train_block) < tau, \
            f"LEAKAGE: train_block max={max(train_block)} >= tau={tau}"

        Xtr_w, ytr_w = stack_prop(dm, train_block)

        g     = dm.graphs[tau]
        m     = g["labeled_mask"]
        Xte_w = g["prop"][m]
        yte_w = g["y"][m].numpy()

        if m.sum() == 0:
            continue

        # Single-class skip guard: can't compute F1 with only one class
        if len(np.unique(yte_w)) < 2:
            continue

        # W8 FIX: compute weights from the expanding window, not static train_steps
        cls_w = _compute_class_weights(ytr_w, device)
        n_epochs = getattr(cfg, "wf_epochs", cfg.sgc_epochs)

        t0 = time.perf_counter()
        model = fit_head(Xtr_w, ytr_w, dm.sgc_input_dim, cfg, cls_w, device, epochs=n_epochs)
        total_wf_train_time += (time.perf_counter() - t0)
        
        model.eval()

        with torch.no_grad():
            s = torch.softmax(model(Xte_w.to(device)), dim=1)[:, 1].cpu().numpy()

        y_pred = (s >= 0.5).astype(int)
        step_f1 = float(f1_score(yte_w, y_pred, pos_label=1, zero_division=0))
        step_prauc = float(average_precision_score(yte_w, s))

        wf_steps.append(tau)
        wf_f1_per_step.append(step_f1)
        wf_prauc_per_step.append(step_prauc)

        y_true_all.append(yte_w)
        s_pred_all.append(s)
        y_pred_all.append(y_pred)
        
        if return_records:
            wf_records.append((tau, step_f1, step_prauc, yte_w, s))

    if wf_steps:
        plt.figure(figsize=(11, 4.5))
        plt.plot(wf_steps, wf_f1_per_step,    marker="o", label="Illicit F1",  color="#C44E52")
        plt.plot(wf_steps, wf_prauc_per_step, marker="s", label="PR-AUC",      color="#4C72B0")
        plt.axvline(cfg.disruption_step, ls="--", color="k",
                    label=f"t={cfg.disruption_step} dark-market shutdown")
        plt.xlabel("Test time step τ")
        plt.ylabel("Score")
        plt.title(
            f"Walk-forward [{safe_name}] "
            f"multiscale={cfg.use_multiscale_prop} "
            f"topo={cfg.use_topology}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # W7 FIX: unique filename per sweep — no overwrites
        out_file = os.path.join(OUTPUT_DIR, f"walk_forward_drift_{safe_name}.png")
        plt.savefig(out_file)
        plt.close()

    if not wf_steps:
        return (0.0, 0.0, []) if return_records else (0.0, 0.0)

    pooled_f1, pooled_prauc, macro_f1, macro_prauc = _aggregate_walk_forward(y_true_all, y_pred_all, s_pred_all)
    
    print(
        f"[{safe_name}] Pooled F1={pooled_f1:.3f} | Macro F1={macro_f1:.3f} | "
        f"Pooled PRAUC={pooled_prauc:.3f} | Macro PRAUC={macro_prauc:.3f}"
    )
    print(f"  [WF Pipeline] Walk-forward explicit training loop overhead: {total_wf_train_time:.2f} seconds")
    return (pooled_f1, pooled_prauc, wf_records) if return_records else (pooled_f1, pooled_prauc)


def walk_forward_validation_adaptive(
    dm: Any,
    cfg: Config,
    device: torch.device,
    delta: float = 0.002,
    sweep_name: str = "adwin",
    return_records: bool = False,
) -> Any:
    """
    Walk-forward validation with ADWIN-adaptive window boundaries.

    Instead of expanding (no window) or fixed sliding (window=W), the training
    window at each test step τ is determined by an ADWIN drift detector
    observing a leakage-safe stream signal computed from data strictly before τ.

    Stream signal (Option A — distribution-shift):
        Per-timestep labeled illicit rate = illicit_count / labeled_count.
        This is computable from training/historical data only, never from
        test labels at τ.

    No-foreknowledge invariant:
        The window boundary for τ depends only on stream values at steps < τ.
        This is enforced by an assertion inside the loop.

    Reuses _aggregate_walk_forward for metric parity with expanding/sliding.

    Args:
        dm:             EllipticDataModule with 'prop' keys set.
        cfg:            Configuration object.
        device:         Torch device.
        delta:          ADWIN confidence parameter (default 0.002).
        sweep_name:     Label for output filenames.
        return_records: If True, return per-step records as third element.

    Returns:
        If return_records:  (pooled_f1, pooled_prauc, wf_records)
        Else:               (pooled_f1, pooled_prauc)
    """
    from evaluation.adwin import ADWIN

    safe_name = re.sub(r"[^\w\-]", "_", sweep_name)

    # ── Step 1: Compute leakage-safe stream ───────────────────────────────
    # Per-timestep labeled illicit rate, for all timesteps in the graph set.
    # This uses only label information at each historical step — never test
    # labels at the current prediction step τ.
    all_ts = sorted(dm.graphs.keys())
    stream_map = {}   # t -> illicit_rate
    for t in all_ts:
        y = dm.graphs[t]["y"].numpy()
        labeled = y[y != -1]
        if len(labeled) > 0:
            stream_map[t] = float((labeled == 1).sum()) / len(labeled)
        else:
            stream_map[t] = 0.0

    # ── Step 2: Replay ADWIN to build the window schedule ─────────────────
    # We replay ADWIN over all timesteps BEFORE the test period to build
    # the initial state, then continue through the test period.
    detector = ADWIN(delta=delta)
    w_start_schedule = {}   # tau -> w_start (as a timestep, not an index)

    for t in all_ts:
        detector.update(stream_map[t])
        # Map the ADWIN internal index back to a timestep
        # detector.w_start is the global index (0-based into all_ts)
        w_start_idx = detector.w_start
        w_start_schedule[t] = all_ts[w_start_idx]

    # ── Step 3: Walk-forward loop (structurally identical to fixed-window) ─
    wf_steps = []
    wf_f1_per_step = []
    wf_prauc_per_step = []
    wf_records = []
    wf_window_widths = []
    y_true_all, y_pred_all, s_pred_all = [], [], []
    total_wf_train_time = 0.0

    for tau in cfg.test_steps:
        # NO-FOREKNOWLEDGE ASSERTION: w_start for tau uses only stream[:tau]
        # The schedule was built by feeding values in order; verify the
        # window start was determined before tau's stream value was relevant.
        adwin_w_start = w_start_schedule.get(tau - 1, min(dm.graphs))
        # The window boundary comes from ADWIN state AFTER consuming up to
        # tau-1 — we look at the schedule at tau-1, not tau, to guarantee
        # no foreknowledge.
        # Actually: we want the state after the detector has seen all values
        # up to and including tau-1. That's w_start_schedule[tau-1].
        # The train block is [adwin_w_start, tau-1].
        if tau - 1 not in w_start_schedule:
            # First test step has no prior history → use all available
            adwin_w_start = min(dm.graphs)

        train_block = list(range(adwin_w_start, tau))
        train_block = [t for t in train_block if t in dm.graphs]

        if not train_block:
            continue

        # LEAKAGE GUARD: no test-step data can be in the training window
        assert max(train_block) < tau, \
            f"LEAKAGE: train_block max={max(train_block)} >= tau={tau}"

        # NO-FOREKNOWLEDGE GUARD: the window start must not depend on
        # any stream value at index >= tau.
        assert adwin_w_start <= tau - 1, \
            f"FOREKNOWLEDGE: w_start={adwin_w_start} > tau-1={tau-1}"

        Xtr_w, ytr_w = stack_prop(dm, train_block)

        g = dm.graphs[tau]
        m = g["labeled_mask"]
        Xte_w = g["prop"][m]
        yte_w = g["y"][m].numpy()

        if m.sum() == 0:
            continue

        # Single-class skip guard
        if len(np.unique(yte_w)) < 2:
            continue

        cls_w = _compute_class_weights(ytr_w, device)
        n_epochs = getattr(cfg, "wf_epochs", cfg.sgc_epochs)

        t0 = time.perf_counter()
        model = fit_head(Xtr_w, ytr_w, dm.sgc_input_dim, cfg, cls_w, device, epochs=n_epochs)
        total_wf_train_time += (time.perf_counter() - t0)

        model.eval()

        with torch.no_grad():
            s = torch.softmax(model(Xte_w.to(device)), dim=1)[:, 1].cpu().numpy()

        y_pred = (s >= 0.5).astype(int)
        step_f1 = float(f1_score(yte_w, y_pred, pos_label=1, zero_division=0))
        step_prauc = float(average_precision_score(yte_w, s))

        wf_steps.append(tau)
        wf_f1_per_step.append(step_f1)
        wf_prauc_per_step.append(step_prauc)
        wf_window_widths.append(len(train_block))

        y_true_all.append(yte_w)
        s_pred_all.append(s)
        y_pred_all.append(y_pred)

        if return_records:
            wf_records.append((tau, step_f1, step_prauc, adwin_w_start, len(train_block)))

    # ── Plot: per-step F1 + window width ──────────────────────────────────
    if wf_steps:
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # F1 on left axis
        ax1.plot(wf_steps, wf_f1_per_step, marker="o", color="#C44E52", label="Illicit F1")
        ax1.plot(wf_steps, wf_prauc_per_step, marker="s", color="#4C72B0", label="PR-AUC")
        ax1.axvline(cfg.disruption_step, ls="--", color="k",
                    label=f"t={cfg.disruption_step} dark-market shutdown")
        ax1.set_xlabel("Test time step τ")
        ax1.set_ylabel("Score")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        # Window width on right axis
        ax2 = ax1.twinx()
        ax2.bar(wf_steps, wf_window_widths, alpha=0.25, color="#55A868", label="ADWIN window width")
        ax2.set_ylabel("Window width (timesteps)")
        ax2.legend(loc="upper right")

        plt.title(
            f"Walk-forward ADWIN [{safe_name}] δ={delta} | "
            f"multiscale={cfg.use_multiscale_prop} topo={cfg.use_topology}"
        )
        plt.tight_layout()

        out_file = os.path.join(OUTPUT_DIR, f"walk_forward_adwin_{safe_name}.png")
        plt.savefig(out_file)
        plt.close()
        print(f"ADWIN walk-forward plot saved to {out_file}")

    if not wf_steps:
        return (0.0, 0.0, []) if return_records else (0.0, 0.0)

    # SHARED AGGREGATION — identical to expanding/sliding evaluators
    pooled_f1, pooled_prauc, macro_f1, macro_prauc = _aggregate_walk_forward(
        y_true_all, y_pred_all, s_pred_all
    )

    print(
        f"[{safe_name}] ADWIN Pooled F1={pooled_f1:.3f} | Macro F1={macro_f1:.3f} | "
        f"Pooled PRAUC={pooled_prauc:.3f} | Macro PRAUC={macro_prauc:.3f}"
    )
    print(f"  [ADWIN] Window widths per τ: {list(zip(wf_steps, wf_window_widths))}")
    print(f"  [ADWIN] Training loop overhead: {total_wf_train_time:.2f} seconds")

    return (pooled_f1, pooled_prauc, wf_records) if return_records else (pooled_f1, pooled_prauc)
