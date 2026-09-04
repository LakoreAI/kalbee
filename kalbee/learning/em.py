"""
Expectation-Maximization (EM) parameter learning for linear-Gaussian
state-space models.

Given a measurement sequence and a fixed transition/measurement structure
(``F``, ``H``), this learns the process- and measurement-noise covariances
``Q`` and ``R`` by maximizing the marginal likelihood — the classic
Shumway & Stoffer (1982) algorithm. Each iteration runs a Kalman filter forward
and an RTS smoother backward (the E-step), then updates ``Q``/``R`` from the
expected sufficient statistics (the M-step). The marginal log-likelihood is
guaranteed non-decreasing across iterations.

This complements the online :class:`~kalbee.AdaptiveKalmanFilter` (innovation-based
adaptation) with an offline, batch maximum-likelihood fit.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from kalbee.modules.utils.linalg import safe_inv


@dataclass
class EMResult:
    """Result of an EM fit."""

    Q: np.ndarray
    R: np.ndarray
    loglik_history: List[float] = field(default_factory=list)
    n_iter_run: int = 0
    converged: bool = False


def _as_measurement_stack(measurements: np.ndarray) -> np.ndarray:
    """Coerce measurements to shape (T, m, 1)."""
    z = np.asarray(measurements, dtype=float)
    if z.ndim == 1:  # (T,) scalar measurements
        z = z.reshape(-1, 1, 1)
    elif z.ndim == 2:  # (T, m)
        z = z.reshape(z.shape[0], z.shape[1], 1)
    return z


def em_kalman(
    measurements: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    n_iter: int = 50,
    learn_Q: bool = True,
    learn_R: bool = True,
    tol: float = 1e-5,
) -> EMResult:
    """
    Learn ``Q`` and/or ``R`` of a linear-Gaussian model by EM.

    Args:
        measurements: Sequence of measurements, shape (T, m) or (T,) for scalar.
        F: State transition matrix (n, n).
        H: Measurement matrix (m, n).
        Q: Initial process-noise covariance (n, n). Defaults to identity.
        R: Initial measurement-noise covariance (m, m). Defaults to identity.
        x0: Initial state mean (n, 1). Defaults to zeros.
        P0: Initial state covariance (n, n). Defaults to identity * 1e2.
        n_iter: Maximum EM iterations.
        learn_Q: Whether to update Q.
        learn_R: Whether to update R.
        tol: Stop when the log-likelihood improves by less than this.

    Returns:
        An :class:`EMResult` with the learned ``Q``/``R`` and the
        (non-decreasing) log-likelihood history.
    """
    z = _as_measurement_stack(measurements)
    T = z.shape[0]
    n = F.shape[0]

    F = np.asarray(F, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.eye(n) if Q is None else np.asarray(Q, dtype=float).copy()
    R = np.eye(H.shape[0]) if R is None else np.asarray(R, dtype=float).copy()
    x0 = np.zeros((n, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(n, 1)
    P0 = np.eye(n) * 1e2 if P0 is None else np.asarray(P0, dtype=float)

    loglik_history: List[float] = []
    converged = False
    iters_run = 0

    for iteration in range(n_iter):
        iters_run = iteration + 1

        # ---- E-step: forward filter ----
        x_pred = [None] * T
        P_pred = [None] * T
        x_filt = [None] * T
        P_filt = [None] * T
        K_last = None
        loglik = 0.0

        x_prev, P_prev = x0, P0
        for t in range(T):
            xp = F @ x_prev if t > 0 else x0
            Pp = (F @ P_prev @ F.T + Q) if t > 0 else P0
            x_pred[t], P_pred[t] = xp, Pp

            y = z[t] - H @ xp
            S = H @ Pp @ H.T + R
            S_inv = safe_inv(S)
            K = Pp @ H.T @ S_inv
            K_last = K

            xf = xp + K @ y
            IKH = np.eye(n) - K @ H
            Pf = IKH @ Pp @ IKH.T + K @ R @ K.T
            x_filt[t], P_filt[t] = xf, (Pf + Pf.T) / 2.0

            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (
                H.shape[0] * np.log(2 * np.pi) + logdet + (y.T @ S_inv @ y).item()
            )

            x_prev, P_prev = xf, Pf

        loglik_history.append(float(loglik))

        # ---- E-step: RTS smoother + lag-one covariances ----
        x_s = [None] * T
        P_s = [None] * T
        G = [None] * T  # smoother gains, G[t] used to go from t+1 -> t
        x_s[T - 1] = x_filt[T - 1]
        P_s[T - 1] = P_filt[T - 1]
        for t in range(T - 2, -1, -1):
            G[t] = P_filt[t] @ F.T @ safe_inv(P_pred[t + 1])
            x_s[t] = x_filt[t] + G[t] @ (x_s[t + 1] - x_pred[t + 1])
            Ps = P_filt[t] + G[t] @ (P_s[t + 1] - P_pred[t + 1]) @ G[t].T
            P_s[t] = (Ps + Ps.T) / 2.0

        # Lag-one smoothed covariances P_lag[t] = Cov(x_t, x_{t-1}), t = 1..T-1
        P_lag = [None] * T
        if T >= 2:
            P_lag[T - 1] = (np.eye(n) - K_last @ H) @ F @ P_filt[T - 2]
            for t in range(T - 2, 0, -1):
                P_lag[t] = (
                    P_filt[t] @ G[t - 1].T
                    + G[t] @ (P_lag[t + 1] - F @ P_filt[t]) @ G[t - 1].T
                )

        # ---- M-step ----
        if learn_R:
            R_new = np.zeros_like(R)
            for t in range(T):
                resid = z[t] - H @ x_s[t]
                R_new += resid @ resid.T + H @ P_s[t] @ H.T
            R_new /= T
            R = (R_new + R_new.T) / 2.0

        if learn_Q and T >= 2:
            A = np.zeros((n, n))  # sum_{t=0}^{T-2}
            B = np.zeros((n, n))  # sum_{t=1}^{T-1} cross
            C = np.zeros((n, n))  # sum_{t=1}^{T-1}
            for t in range(T - 1):
                A += P_s[t] + x_s[t] @ x_s[t].T
            for t in range(1, T):
                C += P_s[t] + x_s[t] @ x_s[t].T
                B += P_lag[t] + x_s[t] @ x_s[t - 1].T
            Q_new = (C - B @ F.T - F @ B.T + F @ A @ F.T) / (T - 1)
            Q = (Q_new + Q_new.T) / 2.0

        # ---- Convergence check ----
        if iteration > 0:
            if abs(loglik_history[-1] - loglik_history[-2]) < tol:
                converged = True
                break

    return EMResult(
        Q=Q,
        R=R,
        loglik_history=loglik_history,
        n_iter_run=iters_run,
        converged=converged,
    )
