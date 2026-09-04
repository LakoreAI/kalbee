"""
Automatic Q/R tuning for Kalman filters using NIS statistics.

Provides methods to automatically tune process noise (Q) and measurement
noise (R) covariances based on the Normalized Innovation Squared (NIS)
statistics. The key insight is that for a correctly tuned filter, the
mean NIS should equal the measurement dimension.

References:
    - Chen, Z., et al. (2019). Kalman filter tuning with Bayesian optimization.
    - Myers, K., & Tapley, B. D. (1976). Adaptive sequential estimation
      with unknown noise statistics.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np

from kalbee.constants import DEFAULT_INITIAL_COVARIANCE
from kalbee.modules.filters.kf_filter import KalmanFilter


@dataclass
class TuneResult:
    """Result of auto-tuning."""
    Q: np.ndarray
    R: np.ndarray
    nis_history: List[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0


def tune_kalman_filter(
    measurements: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    Q_init: Optional[np.ndarray] = None,
    R_init: Optional[np.ndarray] = None,
    n_iter: int = 50,
    learning_rate: float = 0.01,
    target_nis_ratio: float = 1.0,
    tol: float = 1e-4,
) -> TuneResult:
    """
    Auto-tune Q and R for a linear Kalman filter using NIS-based gradient descent.

    The algorithm adjusts Q and R to make the mean NIS equal to the measurement
    dimension (m), which indicates a properly tuned filter.

    Args:
        measurements: Array of shape (T, m) or (T,) for scalar measurements.
        F: State transition matrix (n, n).
        H: Measurement matrix (m, n).
        x0: Initial state (n, 1). Defaults to zeros.
        P0: Initial covariance (n, n). Defaults to 100 * I.
        Q_init: Initial process noise covariance (n, n). Defaults to 0.01 * I.
        R_init: Initial measurement noise covariance (m, m). Defaults to I.
        n_iter: Maximum number of tuning iterations.
        learning_rate: Step size for gradient-based tuning.
        target_nis_ratio: Target ratio of mean_NIS / m (1.0 is ideal).
        tol: Convergence tolerance on NIS ratio change.

    Returns:
        TuneResult with the tuned Q and R matrices.
    """
    z = np.asarray(measurements, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    T, m = z.shape
    n = F.shape[0]

    # Initialize parameters
    x0 = np.zeros((n, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(n, 1)
    P0 = (
        np.eye(n) * DEFAULT_INITIAL_COVARIANCE
        if P0 is None
        else np.asarray(P0, dtype=float)
    )
    Q = np.eye(n) * 0.01 if Q_init is None else np.asarray(Q_init, dtype=float).copy()
    R = np.eye(m) if R_init is None else np.asarray(R_init, dtype=float).copy()

    nis_history = []
    converged = False

    for iteration in range(n_iter):
        # Run filter with current Q, R
        innovations, innovation_covs, mean_nis = _run_filter_and_collect(
            z, F, H, x0, P0, Q, R
        )
        nis_history.append(mean_nis)

        # Compute NIS ratio: mean_NIS / m
        nis_ratio = mean_nis / m

        # Check convergence
        if iteration > 0:
            if abs(nis_history[-1] - nis_history[-2]) < tol:
                converged = True
                break

        # Adjust Q and R based on NIS ratio
        if nis_ratio > target_nis_ratio:
            # NIS too high: filter is overconfident (Q too small or R too small)
            # Increase Q or R
            adjustment = learning_rate * (nis_ratio - target_nis_ratio)
            Q *= (1.0 + adjustment)
            R *= (1.0 + adjustment * 0.5)
        elif nis_ratio < target_nis_ratio:
            # NIS too low: filter is underconfident (Q too large or R too large)
            # Decrease Q or R
            adjustment = learning_rate * (target_nis_ratio - nis_ratio)
            Q *= (1.0 - adjustment)
            R *= (1.0 - adjustment * 0.5)

        # Ensure Q and R stay positive definite
        Q = _ensure_positive_definite(Q)
        R = _ensure_positive_definite(R)

    return TuneResult(
        Q=Q,
        R=R,
        nis_history=nis_history,
        converged=converged,
        iterations=len(nis_history),
    )


def _run_filter_and_collect(
    measurements: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """Run KF and collect innovations for NIS computation."""
    T = measurements.shape[0]

    kf = KalmanFilter(
        state=x0.copy(),
        covariance=P0.copy(),
        transition_matrix=F,
        transition_covariance=Q,
        measurement_matrix=H,
        measurement_covariance=R,
    )

    innovations = []
    innovation_covs = []

    for t in range(T):
        kf.predict()
        kf.update(measurements[t].reshape(-1, 1))

        if hasattr(kf, "last_y") and kf.last_y is not None:
            innovations.append(kf.last_y)
            innovation_covs.append(kf.last_S)

    if len(innovations) == 0:
        return [], [], 0.0

    # Compute mean NIS
    total_nis = 0.0
    for v, S in zip(innovations, innovation_covs):
        S_inv = np.linalg.inv(S)
        total_nis += (v.T @ S_inv @ v).item()
    mean_nis = total_nis / len(innovations)

    return innovations, innovation_covs, mean_nis


def _ensure_positive_definite(M: np.ndarray, min_diag: float = 1e-8) -> np.ndarray:
    """Ensure matrix is positive definite by adjusting diagonal."""
    M = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    if np.any(eigvals <= 0):
        eigvals = np.maximum(eigvals, min_diag)
        M = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return M


def quick_tune(
    measurements: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    process_var: float = 0.01,
    measurement_var: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick single-pass tuning based on innovation statistics.

    This is a faster but less precise alternative to tune_kalman_filter.
    It performs a single filter pass and adjusts Q and R based on the
    observed innovation statistics.

    Args:
        measurements: Array of shape (T, m).
        F: State transition matrix (n, n).
        H: Measurement matrix (m, n).
        process_var: Initial process noise variance.
        measurement_var: Initial measurement noise variance.

    Returns:
        Tuple of (Q, R) covariance matrices.
    """
    z = np.asarray(measurements, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    T, m = z.shape
    n = F.shape[0]

    Q = np.eye(n) * process_var
    R = np.eye(m) * measurement_var

    # Run filter
    _, _, mean_nis = _run_filter_and_collect(
        z, F, H, np.zeros((n, 1)), np.eye(n) * DEFAULT_INITIAL_COVARIANCE, Q, R
    )

    # Scale R to make mean_NIS ≈ m
    if mean_nis > 0:
        scale = mean_nis / m
        R *= scale

    return Q, R
