from typing import Tuple
import numpy as np

from kalbee.modules.utils.linalg import safe_inv


def covariance_intersection(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    omega: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance Intersection (CI) fusion.

    Fuses two estimates with unknown (or potentially correlated) cross-covariance.
    Guarantees a consistent fusion result regardless of the actual correlation.

    The fused estimate is:
        P_fused^{-1} = omega * P_a^{-1} + (1-omega) * P_b^{-1}
        P_fused^{-1} * x_fused = omega * P_a^{-1} * x_a + (1-omega) * P_b^{-1} * x_b

    Args:
        mean_a: State estimate from sensor/filter A (n x 1).
        cov_a: Covariance from sensor/filter A (n x n).
        mean_b: State estimate from sensor/filter B (n x 1).
        cov_b: Covariance from sensor/filter B (n x n).
        omega: Weight for estimate A (0 to 1). omega=0.5 gives equal weight.
               If None, optimal omega is found by minimizing trace(P_fused).

    Returns:
        Tuple of (fused_mean, fused_covariance).
    """
    if omega is None:
        omega = _find_optimal_omega(cov_a, cov_b)

    # Clamp omega to [0, 1]
    omega = max(0.0, min(1.0, omega))

    # Inverse covariances
    inv_a = safe_inv(cov_a)
    inv_b = safe_inv(cov_b)

    # Fused inverse covariance
    inv_fused = omega * inv_a + (1 - omega) * inv_b

    # Fused covariance
    P_fused = safe_inv(inv_fused)

    # Fused mean
    x_fused = P_fused @ (omega * inv_a @ mean_a + (1 - omega) * inv_b @ mean_b)

    # Enforce symmetry
    P_fused = (P_fused + P_fused.T) / 2.0

    return x_fused, P_fused


def _find_optimal_omega(
    cov_a: np.ndarray,
    cov_b: np.ndarray,
    steps: int = 100,
) -> float:
    """
    Find optimal omega that minimizes trace(P_fused).

    Uses a simple grid search over [0, 1].

    Args:
        cov_a: Covariance from A.
        cov_b: Covariance from B.
        steps: Number of grid points.

    Returns:
        Optimal omega value.
    """
    best_omega = 0.5
    best_trace = float("inf")

    for i in range(steps + 1):
        omega = i / steps
        inv_a = safe_inv(cov_a)
        inv_b = safe_inv(cov_b)
        inv_fused = omega * inv_a + (1 - omega) * inv_b
        P_fused = safe_inv(inv_fused)
        tr = np.trace(P_fused)

        if tr < best_trace:
            best_trace = tr
            best_omega = omega

    return best_omega


def sequential_covariance_intersection(
    estimates: list,
    omega: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequential Covariance Intersection for fusing multiple estimates.

    Fuses N estimates by applying CI sequentially.

    Args:
        estimates: List of (mean, covariance) tuples.
        omega: Weight for the "newer" estimate in each pairwise fusion.

    Returns:
        Tuple of (fused_mean, fused_covariance).
    """
    if len(estimates) == 0:
        raise ValueError("No estimates to fuse")

    if len(estimates) == 1:
        return estimates[0]

    # Start with the first estimate
    fused_mean, fused_cov = estimates[0]

    # Sequentially fuse with each subsequent estimate
    for mean_i, cov_i in estimates[1:]:
        fused_mean, fused_cov = covariance_intersection(
            fused_mean,
            fused_cov,
            mean_i,
            cov_i,
            omega=omega,
        )

    return fused_mean, fused_cov
