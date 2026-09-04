"""
Real-time outlier detection using chi-squared tests on NIS.

Provides a configurable outlier detector that monitors the Normalized
Innovation Squared (NIS) and flags measurements as outliers when they
exceed adaptive or fixed thresholds. Useful for real-time tracking systems
where clutter, false alarms, or sensor glitches must be rejected.

References:
    - Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
      Estimation with Applications to Tracking and Navigation.
    - Fortmann, T. E., Bar-Shalom, Y., & Scheffe, M. (1983).
      Sonar tracking of multiple targets using joint probabilistic data association.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2


@dataclass
class DetectionResult:
    """Result of an outlier detection check."""

    is_inlier: bool
    nis_value: float
    threshold: float
    confidence: float
    innovation: Optional[np.ndarray] = None


class Chi2OutlierDetector:
    """
    Real-time outlier detector using chi-squared gating.

    Monitors the NIS of incoming measurements and classifies them as
    inliers or outliers based on a configurable confidence level.

    Features:
    - Fixed threshold mode: Use a fixed chi-squared threshold
    - Adaptive mode: Adjusts threshold based on running NIS statistics
    - Sliding window: Uses recent NIS history for adaptation
    """

    def __init__(
        self,
        m: int,
        confidence: float = 0.95,
        adaptive: bool = False,
        window_size: int = 50,
        scale_factor: float = 3.0,
    ):
        """
        Initialize the outlier detector.

        Args:
            m: Measurement dimension.
            confidence: Confidence level for chi-squared gate (e.g., 0.95).
            adaptive: If True, adapt threshold based on running statistics.
            window_size: Number of recent NIS values for adaptation.
            scale_factor: Multiplier for adaptive threshold (e.g., 3.0 = 3-sigma).
        """
        self.m = m
        self.confidence = confidence
        self.adaptive = adaptive
        self.window_size = window_size
        self.scale_factor = scale_factor

        # Fixed threshold from chi-squared distribution
        self.fixed_threshold = chi2.ppf(confidence, df=m)

        # Adaptive threshold state
        self._nis_buffer: list = []
        self._adaptive_threshold = self.fixed_threshold

    def check(
        self,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> DetectionResult:
        """
        Check if a measurement is an inlier or outlier.

        Args:
            innovation: Innovation vector v = z - H*x_pred (m x 1).
            innovation_covariance: Innovation covariance S (m x m).

        Returns:
            DetectionResult with classification and statistics.
        """
        v = np.asanyarray(innovation, dtype=float).reshape(-1, 1)
        S = np.asanyarray(innovation_covariance, dtype=float)

        # Compute NIS
        S_inv = np.linalg.inv(S)
        nis_value = float((v.T @ S_inv @ v).item())

        # Get threshold
        threshold = self._get_threshold()

        # Classify
        is_inlier = bool(nis_value <= threshold)

        # Always update NIS buffer for statistics
        self._nis_buffer.append(nis_value)
        if len(self._nis_buffer) > self.window_size:
            self._nis_buffer.pop(0)

        # Update adaptive threshold if enabled
        if self.adaptive:
            self._update_adaptive()

        return DetectionResult(
            is_inlier=is_inlier,
            nis_value=nis_value,
            threshold=threshold,
            confidence=self.confidence,
            innovation=innovation.copy(),
        )

    def _get_threshold(self) -> float:
        """Get the current threshold (fixed or adaptive)."""
        if self.adaptive:
            return self._adaptive_threshold
        return self.fixed_threshold

    def _update_adaptive(self):
        """Update adaptive threshold based on recent NIS values."""
        if len(self._nis_buffer) >= 10:
            nis_arr = np.array(self._nis_buffer)
            # Adaptive threshold: mean + scale_factor * std
            mean_nis = np.mean(nis_arr)
            std_nis = np.std(nis_arr)
            self._adaptive_threshold = mean_nis + self.scale_factor * std_nis

    def batch_check(
        self,
        innovations: np.ndarray,
        innovation_covariances: np.ndarray,
    ) -> list:
        """
        Check multiple measurements at once.

        Args:
            innovations: Array of shape (T, m) or list of (m, 1) arrays.
            innovation_covariances: Array of shape (T, m, m) or list of (m, m) arrays.

        Returns:
            List of DetectionResult for each measurement.
        """
        results = []
        T = innovations.shape[0] if hasattr(innovations, "shape") else len(innovations)

        for t in range(T):
            v = (
                innovations[t].reshape(-1, 1)
                if innovations[t].ndim == 1
                else innovations[t]
            )
            S = innovation_covariances[t]
            results.append(self.check(v, S))

        return results

    def get_statistics(self) -> dict:
        """
        Get statistics of the detection process.

        Returns:
            Dictionary with detection statistics.
        """
        if not self._nis_buffer:
            return {"num_checks": 0}

        nis_arr = np.array(self._nis_buffer)
        return {
            "num_checks": len(self._nis_buffer),
            "nis_mean": float(np.mean(nis_arr)),
            "nis_std": float(np.std(nis_arr)),
            "current_threshold": self._get_threshold(),
            "fixed_threshold": self.fixed_threshold,
        }

    def reset(self):
        """Reset the detector state."""
        self._nis_buffer.clear()
        self._adaptive_threshold = self.fixed_threshold
