from typing import Tuple
import numpy as np

from kalbee.modules.utils.linalg import safe_inv


def track_to_track_fusion(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    cross_covariance: np.ndarray = None,
    method: str = "ci",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse two track estimates into a single estimate.

    Args:
        mean_a: State estimate from track A (n x 1).
        cov_a: Covariance from track A (n x n).
        mean_b: State estimate from track B (n x 1).
        cov_b: Covariance from track B (n x n).
        cross_covariance: Cross-covariance between A and B (n x n).
                          If None, assumes uncorrelated.
        method: Fusion method - "ci" (covariance intersection),
                "klf" (Kalman-like fusion), or "simple" (weighted average).

    Returns:
        Tuple of (fused_mean, fused_covariance).
    """
    if method == "ci":
        return _fusion_ci(mean_a, cov_a, mean_b, cov_b)
    elif method == "klf":
        return _fusion_klf(mean_a, cov_a, mean_b, cov_b, cross_covariance)
    elif method == "simple":
        return _fusion_simple(mean_a, cov_a, mean_b, cov_b)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def _fusion_ci(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Covariance Intersection fusion (consistent for unknown correlations)."""
    from kalbee.modules.fusion.covariance_intersection import covariance_intersection

    return covariance_intersection(mean_a, cov_a, mean_b, cov_b, omega=None)


def _fusion_klf(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    cross_covariance: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman-like fusion (optimal if tracks are uncorrelated).

    If cross_covariance is provided, accounts for correlation.
    """
    if cross_covariance is None:
        # Assume uncorrelated
        P_a_inv = safe_inv(cov_a)
        P_b_inv = safe_inv(cov_b)

        P_fused = safe_inv(P_a_inv + P_b_inv)
        x_fused = P_fused @ (P_a_inv @ mean_a + P_b_inv @ mean_b)
    else:
        # Account for correlation (Bar-Shalom / Campo formula)
        P_ab = cross_covariance
        P_ba = cross_covariance.T

        S_inv = safe_inv(cov_a + cov_b - P_ab - P_ba)
        gain = (cov_a - P_ab) @ S_inv
        x_fused = mean_a + gain @ (mean_b - mean_a)
        P_fused = cov_a - gain @ (cov_a - P_ba)

    P_fused = (P_fused + P_fused.T) / 2.0
    return x_fused, P_fused


def _fusion_simple(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple weighted average based on inverse covariance."""
    P_a_inv = safe_inv(cov_a)
    P_b_inv = safe_inv(cov_b)

    # Weight by precision (inverse variance)
    w_a = np.trace(P_a_inv)
    w_b = np.trace(P_b_inv)
    w_total = w_a + w_b

    x_fused = (w_a * mean_a + w_b * mean_b) / w_total
    P_fused = safe_inv(P_a_inv + P_b_inv)
    P_fused = (P_fused + P_fused.T) / 2.0

    return x_fused, P_fused
