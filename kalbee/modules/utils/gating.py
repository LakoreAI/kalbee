"""
Innovation gating and outlier rejection utilities for Kalman filters.

Provides chi-squared based validation gating to reject outlier measurements
before they corrupt the filter state. This is critical for real-world tracking
where clutter, false alarms, and sensor glitches are common.

References:
    - Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
      Estimation with Applications to Tracking and Navigation.
    - Fortmann, T. E., Bar-Shalom, Y., & Scheffe, M. (1983).
      Sonar tracking of multiple targets using joint probabilistic data association.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.stats import chi2


def nis(innovation: np.ndarray, innovation_covariance: np.ndarray) -> float:
    """
    Compute Normalized Innovation Squared (NIS).

    NIS = v^T S^{-1} v

    where v is the innovation and S is the innovation covariance.

    Under correct filter tuning, NIS follows a chi-squared distribution
    with m degrees of freedom (m = measurement dimension).

    Args:
        innovation: Innovation vector (m x 1).
        innovation_covariance: Innovation covariance matrix S (m x m).

    Returns:
        Scalar NIS value.
    """
    v = np.asarray(innovation, dtype=float).reshape(-1, 1)
    S = np.asarray(innovation_covariance, dtype=float)

    S_inv = np.linalg.inv(S)
    return float((v.T @ S_inv @ v).item())


def chi2_gate(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[bool, float, float]:
    """
    Chi-squared validation gate for outlier rejection.

    Tests whether a measurement innovation is consistent with the filter's
    predicted innovation covariance at the given confidence level.

    Args:
        innovation: Innovation vector v = z - H*x_pred (m x 1).
        innovation_covariance: Innovation covariance S = H*P*H' + R (m x m).
        confidence: Confidence level for the gate (e.g., 0.95 for 95%).

    Returns:
        Tuple of (passed, nis_value, threshold):
        - passed: True if the measurement passes the gate (not an outlier).
        - nis_value: The computed NIS value.
        - threshold: The chi-squared threshold at the given confidence level.
    """
    m = innovation.shape[0] if innovation.ndim > 0 else 1
    nis_value = nis(innovation, innovation_covariance)
    threshold = chi2.ppf(confidence, df=m)
    passed = bool(nis_value <= threshold)
    return passed, nis_value, threshold


def mahalanobis_distance(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
) -> float:
    """
    Compute the Mahalanobis distance of an innovation.

    This is the square root of NIS, giving a distance in standard deviation units.

    Args:
        innovation: Innovation vector (m x 1).
        innovation_covariance: Innovation covariance matrix S (m x m).

    Returns:
        Scalar Mahalanobis distance.
    """
    return np.sqrt(nis(innovation, innovation_covariance))


def ellipsoidal_gate(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
    gate_threshold: float = 5.0,
) -> bool:
    """
    Ellipsoidal gating using Mahalanobis distance.

    A simple and common gating method where measurements are accepted if
    the Mahalanobis distance is below a fixed threshold.

    Args:
        innovation: Innovation vector (m x 1).
        innovation_covariance: Innovation covariance matrix S (m x m).
        gate_threshold: Maximum allowed Mahalanobis distance.

    Returns:
        True if the measurement passes the gate.
    """
    dist = mahalanobis_distance(innovation, innovation_covariance)
    return bool(dist <= gate_threshold)


def gated_update(
    filter_obj,
    measurement: np.ndarray,
    confidence: float = 0.95,
    gate_threshold: Optional[float] = None,
) -> Tuple[bool, np.ndarray]:
    """
    Perform a gated filter update. Rejects outlier measurements.

    Args:
        filter_obj: A BaseFilter instance (must have last_y, last_S, or
                    measurement_matrix, measurement_covariance attributes).
        measurement: The measurement to potentially update with (m x 1).
        confidence: Confidence level for chi-squared gating (used if
                    gate_threshold is None).
        gate_threshold: If provided, use Mahalanobis distance gating
                        instead of chi-squared gating.

    Returns:
        Tuple of (updated, state):
        - updated: True if the measurement was accepted and update performed.
        - state: The current state (updated or not).
    """
    z = np.asarray(measurement, dtype=float).reshape(-1, 1)

    # Compute innovation and innovation covariance
    last_y = getattr(filter_obj, "last_y", None)
    last_S = getattr(filter_obj, "last_S", None)

    if last_y is not None and last_S is not None:
        y = last_y
        S = last_S
    else:
        H = getattr(filter_obj, "measurement_matrix", None)
        R = getattr(filter_obj, "measurement_covariance", None)
        if H is None or R is None:
            # Cannot compute gate, perform update anyway
            filter_obj.update(z)
            return True, filter_obj.state
        y = z - H @ filter_obj.state
        S = H @ filter_obj.covariance @ H.T + R

    # Apply gate
    if gate_threshold is not None:
        passed = ellipsoidal_gate(y, S, gate_threshold)
    else:
        passed, _, _ = chi2_gate(y, S, confidence)

    if passed:
        filter_obj.update(z)
        return True, filter_obj.state
    else:
        # Measurement rejected - do not update filter
        return False, filter_obj.state
