"""
Experiment results container.

Stores and summarizes filter tracking results for easy analysis.
"""

from typing import Dict, List, Optional
import numpy as np

from kalbee.modules.utils.metrics import rmse, nees


class ExperimentResult:
    """
    Result from a single filter run on a signal.

    Attributes:
        filter_name: Name of the filter used.
        time_steps: Array of time steps.
        true_states: Ground-truth states, shape (T, n, 1).
        estimated_states: Filter-estimated states, shape (T, n, 1).
        measurements: Noisy measurements, shape (T, m, 1).
        covariances: Filter covariances at each step, list of (n, n).
    """

    def __init__(
        self,
        filter_name: str,
        time_steps: np.ndarray,
        true_states: np.ndarray,
        estimated_states: np.ndarray,
        measurements: np.ndarray,
        covariances: List[np.ndarray],
    ):
        self.filter_name = filter_name
        self.time_steps = time_steps
        self.true_states = true_states
        self.estimated_states = estimated_states
        self.measurements = measurements
        self.covariances = covariances

    def position_rmse(self) -> float:
        """RMSE for the position (first state) component."""
        est_pos = self.estimated_states[:, 0, 0]
        true_pos = self.true_states[:, 0, 0]
        return rmse(est_pos, true_pos)

    def velocity_rmse(self) -> float:
        """RMSE for the velocity (second state) component."""
        est_vel = self.estimated_states[:, 1, 0]
        true_vel = self.true_states[:, 1, 0]
        return rmse(est_vel, true_vel)

    def nees_values(self) -> np.ndarray:
        """Compute NEES at each time step."""
        state_errors = []
        for k in range(len(self.time_steps)):
            e = self.true_states[k] - self.estimated_states[k]
            state_errors.append(e)
        return nees(state_errors, self.covariances)

    def average_nees(self) -> float:
        """Average NEES (should be ≈ state dimension for consistent filter)."""
        return float(np.mean(self.nees_values()))

    def summary_dict(self) -> dict:
        """Return a summary dictionary of this result."""
        return {
            "filter": self.filter_name,
            "position_rmse": round(self.position_rmse(), 6),
            "velocity_rmse": round(self.velocity_rmse(), 6),
            "avg_nees": round(self.average_nees(), 4),
        }


class ExperimentReport:
    """
    Collection of results from running multiple filters on the same signal.

    Provides comparison and summary utilities.
    """

    def __init__(
        self,
        signal_name: str,
        results: List[ExperimentResult],
        signal_params: Optional[dict] = None,
    ):
        self.signal_name = signal_name
        self.results = results
        self.signal_params = signal_params or {}

    def summary(self) -> str:
        """
        Generate a formatted text summary comparing all filters.

        Returns:
            Multi-line string with comparison table.
        """
        lines = []
        lines.append(f"{'=' * 64}")
        lines.append(f"  Experiment Report: {self.signal_name}")
        lines.append(f"{'=' * 64}")

        if self.signal_params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.signal_params.items())
            lines.append(f"  Signal params: {param_str}")
            lines.append("")

        # Table header
        lines.append(
            f"  {'Filter':<25} {'Pos RMSE':>10} {'Vel RMSE':>10} {'Avg NEES':>10}"
        )
        lines.append(f"  {'-' * 55}")

        # Sort by position RMSE (best first)
        sorted_results = sorted(self.results, key=lambda r: r.position_rmse())

        for result in sorted_results:
            s = result.summary_dict()
            lines.append(
                f"  {s['filter']:<25} {s['position_rmse']:>10.4f} "
                f"{s['velocity_rmse']:>10.4f} {s['avg_nees']:>10.4f}"
            )

        lines.append(f"{'=' * 64}")

        # Best filter
        best = sorted_results[0].filter_name
        lines.append(f"  Best position tracking: {best}")
        lines.append("")

        return "\n".join(lines)

    def get_best_filter(self) -> ExperimentResult:
        """Return the result with the lowest position RMSE."""
        return min(self.results, key=lambda r: r.position_rmse())

    def to_dict(self) -> Dict:
        """Convert the full report to a dictionary."""
        return {
            "signal": self.signal_name,
            "params": self.signal_params,
            "results": [r.summary_dict() for r in self.results],
            "best_filter": self.get_best_filter().filter_name,
        }
