"""
Filter performance metrics and diagnostics.

Provides standard metrics for evaluating state estimation filter performance:
- RMSE: Root Mean Square Error
- NEES: Normalized Estimation Error Squared (filter consistency)
- NIS: Normalized Innovation Squared (measurement consistency)
- Log-Likelihood: for model comparison and adaptive filtering
"""

from typing import List
import numpy as np


def rmse(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> float:
    """
    Compute Root Mean Square Error between estimated and true states.

    Args:
        estimated: Estimated states, shape (T, n) or (T, n, 1).
        truth: True states, same shape as estimated.

    Returns:
        Scalar RMSE value.
    """
    est = np.asanyarray(estimated).squeeze()
    tru = np.asanyarray(truth).squeeze()

    if est.ndim == 1:
        est = est.reshape(-1, 1)
        tru = tru.reshape(-1, 1)

    errors = est - tru
    return float(np.sqrt(np.mean(errors**2)))


def nees(
    state_errors: List[np.ndarray],
    covariances: List[np.ndarray],
) -> np.ndarray:
    """
    Compute Normalized Estimation Error Squared (NEES) for filter consistency.

    For a consistent filter, NEES should follow a chi-squared distribution
    with n degrees of freedom (where n is the state dimension).

    The average NEES over time should be approximately n.

    Args:
        state_errors: List of state error vectors (x_true - x_est), each (n x 1).
        covariances: List of corresponding covariance matrices, each (n x n).

    Returns:
        Array of NEES values, one per time step.
    """
    T = len(state_errors)
    nees_values = np.zeros(T)

    for k in range(T):
        e = np.asanyarray(state_errors[k]).reshape(-1, 1)
        P = np.asanyarray(covariances[k])
        P_inv = np.linalg.inv(P)
        nees_values[k] = (e.T @ P_inv @ e).item()

    return nees_values


def nis(
    innovations: List[np.ndarray],
    innovation_covariances: List[np.ndarray],
) -> np.ndarray:
    """
    Compute Normalized Innovation Squared (NIS) for measurement consistency.

    For a correctly tuned filter, NIS should follow a chi-squared distribution
    with m degrees of freedom (where m is the measurement dimension).

    Args:
        innovations: List of innovation vectors (z - H@x), each (m x 1).
        innovation_covariances: List of corresponding S matrices, each (m x m).

    Returns:
        Array of NIS values, one per time step.
    """
    T = len(innovations)
    nis_values = np.zeros(T)

    for k in range(T):
        v = np.asanyarray(innovations[k]).reshape(-1, 1)
        S = np.asanyarray(innovation_covariances[k])
        S_inv = np.linalg.inv(S)
        nis_values[k] = (v.T @ S_inv @ v).item()

    return nis_values


def log_likelihood(
    innovations: List[np.ndarray],
    innovation_covariances: List[np.ndarray],
) -> float:
    """
    Compute the total log-likelihood of the filter given an innovation sequence.

    Useful for model comparison and parameter tuning.

    L = -0.5 * sum_k [ m*log(2*pi) + log(det(S_k)) + v_k^T S_k^{-1} v_k ]

    Args:
        innovations: List of innovation vectors (z - H@x), each (m x 1).
        innovation_covariances: List of corresponding S matrices, each (m x m).

    Returns:
        Total log-likelihood (scalar).
    """
    total = 0.0

    for k in range(len(innovations)):
        v = np.asanyarray(innovations[k]).reshape(-1, 1)
        S = np.asanyarray(innovation_covariances[k])
        m = v.shape[0]

        sign, logdet = np.linalg.slogdet(S)
        S_inv = np.linalg.inv(S)

        total += -0.5 * (m * np.log(2 * np.pi) + logdet + (v.T @ S_inv @ v).item())

    return float(total)
