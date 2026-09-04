"""
Real-time filter diagnostics and monitoring.

Provides a FilterDiagnostics class that collects and reports filter statistics
during operation, including NIS, NEES, innovation statistics, and convergence
metrics. Useful for debugging, tuning, and performance monitoring.

References:
    - Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
      Estimation with Applications to Tracking and Navigation.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class FilterSnapshot:
    """Snapshot of filter state at a single time step."""

    timestamp: int
    state_mean: np.ndarray
    state_cov_trace: float
    innovation: Optional[np.ndarray] = None
    innovation_cov: Optional[np.ndarray] = None
    nis: Optional[float] = None
    nees: Optional[float] = None
    kalman_gain_norm: Optional[float] = None


class FilterDiagnostics:
    """
    Real-time diagnostics collector for Kalman filters.

    Tracks key performance metrics over time:
    - NIS (Normalized Innovation Squared) for measurement consistency
    - NEES (Normalized Estimation Error Squared) for state consistency
    - Innovation statistics (mean, variance)
    - Covariance trace (uncertainty level)
    - Kalman gain norm (filter aggressiveness)

    Usage:
        diag = FilterDiagnostics(m=1, n=2)
        for t in range(T):
            kf.predict()
            kf.update(z[t])
            diag.collect(kf, z[t])

        report = diag.summary()
        diag.plot()
    """

    def __init__(self, m: int, n: int, alpha: float = 0.05):
        """
        Initialize diagnostics collector.

        Args:
            m: Measurement dimension.
            n: State dimension.
            alpha: Significance level for consistency tests.
        """
        self.m = m
        self.n = n
        self.alpha = alpha

        # Storage
        self.snapshots: List[FilterSnapshot] = []
        self.nis_history: List[float] = []
        self.nees_history: List[float] = []
        self.innovation_history: List[np.ndarray] = []
        self.cov_trace_history: List[float] = []
        self.state_history: List[np.ndarray] = []

        # Ground truth (optional, for NEES)
        self._ground_truth: Optional[np.ndarray] = None

    def collect(
        self,
        filter_obj,
        measurement: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
    ) -> FilterSnapshot:
        """
        Collect diagnostics from the current filter state.

        Args:
            filter_obj: A BaseFilter instance.
            measurement: The measurement used in the last update (optional).
            ground_truth: The true state for NEES computation (optional).

        Returns:
            FilterSnapshot with current metrics.
        """
        timestamp = len(self.snapshots)

        # Extract filter state
        state_mean = filter_obj.state.copy()
        state_cov_trace = np.trace(filter_obj.covariance)

        # Innovation and NIS
        innovation = None
        innovation_cov = None
        nis_val = None

        last_y = getattr(filter_obj, "last_y", None)
        last_S = getattr(filter_obj, "last_S", None)

        if last_y is not None and last_S is not None:
            innovation = last_y.copy()
            innovation_cov = last_S.copy()
            self.innovation_history.append(innovation)

            # Compute NIS
            S_inv = np.linalg.inv(last_S)
            nis_val = float((last_y.T @ S_inv @ last_y).item())
            self.nis_history.append(nis_val)

        # NEES (requires ground truth)
        nees_val = None
        if ground_truth is not None:
            error = ground_truth.reshape(-1, 1) - state_mean
            P_inv = np.linalg.inv(filter_obj.covariance)
            nees_val = float((error.T @ P_inv @ error).item())
            self.nees_history.append(nees_val)

        # Kalman gain norm
        kalman_gain_norm = None
        if hasattr(filter_obj, "K") and filter_obj.K is not None:
            kalman_gain_norm = float(np.linalg.norm(filter_obj.K))

        # Store history
        self.cov_trace_history.append(state_cov_trace)
        self.state_history.append(state_mean.copy())

        # Create snapshot
        snapshot = FilterSnapshot(
            timestamp=timestamp,
            state_mean=state_mean,
            state_cov_trace=state_cov_trace,
            innovation=innovation,
            innovation_cov=innovation_cov,
            nis=nis_val,
            nees=nees_val,
            kalman_gain_norm=kalman_gain_norm,
        )
        self.snapshots.append(snapshot)

        return snapshot

    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary report of collected diagnostics.

        Returns:
            Dictionary with summary statistics.
        """
        if len(self.nis_history) == 0:
            return {"message": "No data collected yet"}

        nis_arr = np.array(self.nis_history)
        cov_trace_arr = np.array(self.cov_trace_history)

        report = {
            "num_steps": len(self.snapshots),
            "nis_mean": float(np.mean(nis_arr)),
            "nis_std": float(np.std(nis_arr)),
            "nis_expected": float(self.m),  # Should be m for consistent filter
            "cov_trace_final": float(cov_trace_arr[-1]),
            "cov_trace_mean": float(np.mean(cov_trace_arr)),
        }

        # NIS consistency test
        from kalbee.modules.utils.consistency import nis_test

        if len(self.innovation_history) >= 2:
            passed, _, mean_nis, expected, p_val = nis_test(
                self.innovation_history,
                [
                    s.innovation_cov
                    for s in self.snapshots
                    if s.innovation_cov is not None
                ],
                alpha=self.alpha,
            )
            report["nis_test_passed"] = passed
            report["nis_test_p_value"] = p_val

        # NEES summary if available
        if len(self.nees_history) > 0:
            nees_arr = np.array(self.nees_history)
            report["nees_mean"] = float(np.mean(nees_arr))
            report["nees_std"] = float(np.std(nees_arr))
            report["nees_expected"] = float(self.n)

        return report

    def check_consistency(self) -> Dict[str, bool]:
        """
        Check if the filter is consistent based on collected diagnostics.

        Returns:
            Dictionary with consistency check results.
        """
        results = {}

        if len(self.innovation_history) < 2:
            return {"insufficient_data": True}

        from kalbee.modules.utils.consistency import nis_test

        passed, _, mean_nis, _, p_val = nis_test(
            self.innovation_history,
            [s.innovation_cov for s in self.snapshots if s.innovation_cov is not None],
            alpha=self.alpha,
        )
        results["nis_consistent"] = passed
        results["nis_mean"] = mean_nis

        # Check if NIS is within acceptable range
        nis_arr = np.array(self.nis_history)
        results["nis_in_range"] = bool(
            np.percentile(nis_arr, 5) < self.m * 3
            and np.percentile(nis_arr, 95) > self.m * 0.3
        )

        return results

    def get_innovations(self) -> np.ndarray:
        """Return stacked innovation array (T, m)."""
        if not self.innovation_history:
            return np.array([])
        return np.array([v.flatten() for v in self.innovation_history])

    def get_nis_values(self) -> np.ndarray:
        """Return NIS history as array."""
        return np.array(self.nis_history)

    def reset(self):
        """Clear all collected data."""
        self.snapshots.clear()
        self.nis_history.clear()
        self.nees_history.clear()
        self.innovation_history.clear()
        self.cov_trace_history.clear()
        self.state_history.clear()
