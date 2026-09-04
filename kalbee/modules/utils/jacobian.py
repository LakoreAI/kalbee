"""
Numerical Jacobian utilities for EKF-style filters.

Deriving analytic Jacobians by hand is the single biggest source of friction
when setting up an :class:`~kalbee.ExtendedKalmanFilter`. :func:`numerical_jacobian`
computes them via central finite differences instead, so
``transition_jacobian``/``measurement_jacobian`` can be built directly from
``transition_function``/``measurement_function`` with no extra derivation.
"""

from typing import Callable
import numpy as np


def numerical_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Central-difference Jacobian of ``func`` at ``x``.

    Args:
        func: Vector-valued function mapping an (n, 1) state to an (m, 1) output.
        x: Point (n, 1) at which to evaluate the Jacobian.
        eps: Finite-difference step size.

    Returns:
        Jacobian matrix of shape (m, n).
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    f0 = np.asarray(func(x)).reshape(-1, 1)
    m = f0.shape[0]

    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i, 0] = eps
        f_plus = np.asarray(func(x + dx)).reshape(-1, 1)
        f_minus = np.asarray(func(x - dx)).reshape(-1, 1)
        J[:, i] = ((f_plus - f_minus) / (2 * eps)).flatten()

    return J


def numerical_transition_jacobian(
    f: Callable[[np.ndarray, float], np.ndarray],
    x: np.ndarray,
    dt: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Jacobian of a transition function ``f(x, dt)`` w.r.t. ``x``, via finite
    differences. Drop-in for ``ExtendedKalmanFilter``'s ``F``/``transition_jacobian``.
    """
    return numerical_jacobian(lambda s: f(s, dt), x, eps=eps)


def numerical_measurement_jacobian(
    h: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Jacobian of a measurement function ``h(x)`` w.r.t. ``x``, via finite
    differences. Drop-in for ``ExtendedKalmanFilter``'s ``H``/``measurement_jacobian``.
    """
    return numerical_jacobian(h, x, eps=eps)
