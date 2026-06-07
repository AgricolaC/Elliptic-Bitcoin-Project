"""
adwin.py — Exact-split ADWIN drift detector.

Implements ADWIN (Bifet & Gavaldà, 2007) using exact split-point testing.
This is appropriate for short streams (~49 elements) where the bucketed
histogram variant is unnecessary overhead.

The algorithm maintains a growing window of streamed scalar values. At each
new observation it tests all possible split points: if any split yields
sub-windows whose means differ more than a Hoeffding-bound threshold,
the older sub-window is dropped (drift detected).
"""

import math
import numpy as np
from typing import List


class ADWIN:
    """
    Exact-split ADWIN drift detector for short streams.

    Parameters
    ----------
    delta : float
        Confidence parameter for the Hoeffding bound.  Smaller δ → fewer
        false positives (less sensitive).  Literature default: 0.002.
    """

    def __init__(self, delta: float = 0.002):
        assert 0 < delta < 1, f"delta must be in (0, 1), got {delta}"
        self._delta = delta
        self._window: List[float] = []
        self._total_seen: int = 0   # how many values have ever been consumed

    # ── public interface ──────────────────────────────────────────────────

    def update(self, value: float) -> bool:
        """
        Append *value* to the window and run drift detection.

        Returns True if drift was detected (older data dropped).
        Asserts that the value is finite and the window never empties.
        """
        assert math.isfinite(value), f"ADWIN received non-finite value: {value}"
        self._window.append(value)
        self._total_seen += 1

        drift = self._check_and_cut()

        # INVARIANT: window must never become empty after an update
        assert len(self._window) > 0, "ADWIN window is empty after update"
        return drift

    @property
    def width(self) -> int:
        """Current window length."""
        return len(self._window)

    @property
    def w_start(self) -> int:
        """
        Global index of the window's left boundary.

        If 10 values have been consumed and the current window holds 4,
        then w_start = 6  (the window covers indices [6, 7, 8, 9]).
        """
        return self._total_seen - len(self._window)

    # ── internals ─────────────────────────────────────────────────────────

    # Minimum window size before split-testing begins.
    # With fewer elements, the Hoeffding bound is too loose and
    # random variation triggers false drifts.
    _MIN_WINDOW_SIZE = 5

    def _check_and_cut(self) -> bool:
        """
        Test every split point in the current window.  If the largest
        mean-difference exceeds the Hoeffding bound, drop the older
        sub-window.  Repeat until no further drift is found.

        Split test for sub-windows W0 (older, length n0) and W1 (newer, n1):
            ε_cut = sqrt( (1 / (2·m)) · ln(4 / δ') )
        where m = harmonic mean of n0 and n1
              δ' = δ / |W|    (Bonferroni correction over all split points)
        """
        found_drift = False

        while True:
            n = len(self._window)
            if n < self._MIN_WINDOW_SIZE:
                break

            delta_prime = self._delta / n
            best_cut_idx = -1
            best_excess = 0.0

            # running sums for efficient mean computation
            total_sum = sum(self._window)
            sum_left = 0.0
            for i in range(n - 1):
                sum_left += self._window[i]
                n0 = i + 1
                n1 = n - n0

                # Require each sub-window to have at least 2 elements
                if n0 < 2 or n1 < 2:
                    continue

                mean0 = sum_left / n0
                mean1 = (total_sum - sum_left) / n1

                # harmonic mean of sub-window lengths
                m_harm = (2.0 * n0 * n1) / (n0 + n1)
                eps_cut = math.sqrt((1.0 / (2.0 * m_harm)) * math.log(4.0 / delta_prime))

                excess = abs(mean0 - mean1) - eps_cut
                if excess > best_excess:
                    best_excess = excess
                    best_cut_idx = i

            if best_cut_idx >= 0:
                # Drop older sub-window [0, best_cut_idx]
                self._window = self._window[best_cut_idx + 1:]
                found_drift = True
            else:
                break

        return found_drift


def adwin_window_schedule(
    stream: np.ndarray,
    delta: float = 0.002,
) -> List[int]:
    """
    Given a pre-computed stream of per-timestep scalars, return the window-start
    index that ADWIN would choose at each step τ using only values ``stream[:τ]``.

    Parameters
    ----------
    stream : 1-D array of length T
        Per-timestep scalar statistic (e.g. labeled illicit rate).
    delta : float
        ADWIN confidence parameter.

    Returns
    -------
    schedule : list of int, length T
        ``schedule[τ]`` is the global start index of the ADWIN window after
        consuming ``stream[:τ+1]``.  At test step τ, the training window
        should be ``[schedule[τ], τ)``.

    Notes
    -----
    **No-foreknowledge invariant**: the schedule for index τ depends only on
    ``stream[:τ+1]``.  This is enforced by construction — values are fed to
    ADWIN one at a time in order.
    """
    stream = np.asarray(stream, dtype=float)
    assert stream.ndim == 1, f"stream must be 1-D, got shape {stream.shape}"
    assert np.all(np.isfinite(stream)), "stream contains non-finite values"

    detector = ADWIN(delta=delta)
    schedule: List[int] = []

    for i in range(len(stream)):
        detector.update(stream[i])
        schedule.append(detector.w_start)

    return schedule
