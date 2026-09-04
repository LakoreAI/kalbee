"""
Filter consistency tests using NIS and NEES statistics.

Provides formal hypothesis testing to verify whether a Kalman filter is
properly tuned. A consistent filter should pass both NIS and NEES tests.

References:
    - Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
      Estimation with Applications to Tracking and Navigation.
    - Chen, Z., et al. (2019). Kalman filter tuning with Bayesian optimization.
"""

from typing import Tuple, List
import numpy as np


def nis_test(
    innovations: List[np.ndarray],
    innovation_covariances: List[np.ndarray],
    alpha: float = 0.05,
) -> Tuple[bool, np.ndarray, float, float, float]:
    """
    Perform NIS (Normalized Innovation Squared) consistency test.

    Tests whether the innovation sequence is consistent with the filter's
    predicted innovation covariance. Under correct tuning, NIS values should
    follow a chi-squared distribution with m degrees of freedom.

    Hypothesis:
        H0: The filter is correctly tuned (innovations are consistent)
        H1: The filter is incorrectly tuned

    The test accepts H0 if the fraction of NIS values within the acceptance
    region is close to the expected (1 - alpha) level.

    Args:
        innovations: List of innovation vectors, each (m x 1).
        innovation_covariances: List of corresponding S matrices, each (m x m).
        alpha: Significance level (default 0.05 for 95% confidence).

    Returns:
        Tuple of (passed, nis_values, mean_nis, expected_mean, p_value):
        - passed: True if the filter passes the consistency test
        - nis_values: Array of NIS values
        - mean_nis: Mean of NIS values
        - expected_mean: Expected mean (= m, measurement dimension)
        - p_value: p-value of the chi-squared goodness-of-fit test
    """
    if len(innovations) < 2:
        raise ValueError("Need at least 2 innovation samples for consistency test")

    m = innovations[0].shape[0]
    T = len(innovations)

    nis_values = np.zeros(T)
    for k in range(T):
        v = np.asarray(innovations[k]).reshape(-1, 1)
        S = np.asarray(innovation_covariances[k])
        S_inv = np.linalg.inv(S)
        nis_values[k] = (v.T @ S_inv @ v).item()

    mean_nis = np.mean(nis_values)
    expected_mean = float(m)

    # Chi-squared goodness-of-fit test
    # Under H0, NIS values should be chi-squared distributed with m dof
    # We test if the sample mean is consistent with the expected mean
    # Using the normalized statistic: sqrt(T) * (mean_nis - m) / sqrt(2*m)
    # which is approximately N(0,1) for large T
    std_nis = np.sqrt(2.0 * m / T)
    z_stat = (mean_nis - m) / std_nis

    # Two-sided p-value
    p_value = 2.0 * (1.0 - np.abs(np.clip(z_stat, -6, 6)) / 6.0)
    # More precise: use survival function
    from scipy.stats import norm

    p_value = 2.0 * norm.sf(np.abs(z_stat))

    # Accept H0 if p-value > alpha
    passed = p_value > alpha

    return passed, nis_values, mean_nis, expected_mean, p_value


def nees_test(
    state_errors: List[np.ndarray],
    covariances: List[np.ndarray],
    alpha: float = 0.05,
) -> Tuple[bool, np.ndarray, float, float, float]:
    """
    Perform NEES (Normalized Estimation Error Squared) consistency test.

    Tests whether the state estimation errors are consistent with the
    filter's estimated covariance. Under correct tuning, NEES values should
    follow a chi-squared distribution with n degrees of freedom.

    NOTE: This test requires access to the true state (ground truth).

    Args:
        state_errors: List of state error vectors (x_true - x_est), each (n x 1).
        covariances: List of corresponding covariance matrices, each (n x n).
        alpha: Significance level (default 0.05 for 95% confidence).

    Returns:
        Tuple of (passed, nees_values, mean_nees, expected_mean, p_value):
        - passed: True if the filter passes the consistency test
        - nees_values: Array of NEES values
        - mean_nees: Mean of NEES values
        - expected_mean: Expected mean (= n, state dimension)
        - p_value: p-value of the chi-squared goodness-of-fit test
    """
    if len(state_errors) < 2:
        raise ValueError("Need at least 2 error samples for consistency test")

    n = state_errors[0].shape[0]
    T = len(state_errors)

    nees_values = np.zeros(T)
    for k in range(T):
        e = np.asarray(state_errors[k]).reshape(-1, 1)
        P = np.asarray(covariances[k])
        P_inv = np.linalg.inv(P)
        nees_values[k] = (e.T @ P_inv @ e).item()

    mean_nees = np.mean(nees_values)
    expected_mean = float(n)

    # Chi-squared goodness-of-fit test (same as NIS)
    from scipy.stats import norm

    std_nees = np.sqrt(2.0 * n / T)
    z_stat = (mean_nees - n) / std_nees
    p_value = 2.0 * norm.sf(np.abs(z_stat))

    passed = p_value > alpha

    return passed, nees_values, mean_nees, expected_mean, p_value


def innovation_whiteness_test(
    innovations: List[np.ndarray],
    max_lag: int = 10,
    alpha: float = 0.05,
) -> Tuple[bool, np.ndarray]:
    """
    Test whether the innovation sequence is white (uncorrelated).

    A well-tuned filter should produce white innovations. This test
    computes the autocorrelation of the innovations and checks if it
    is within the confidence bounds.

    Args:
        innovations: List of innovation vectors, each (m x 1).
        max_lag: Maximum lag to test.
        alpha: Significance level for confidence bounds.

    Returns:
        Tuple of (passed, autocorrelations):
        - passed: True if innovations are white (within confidence bounds)
        - autocorrelations: Autocorrelation values at each lag
    """
    if len(innovations) < max_lag + 2:
        raise ValueError(f"Need at least {max_lag + 2} samples for whiteness test")

    m = innovations[0].shape[0]
    T = len(innovations)

    # Stack innovations into matrix (T, m)
    V = np.array([v.flatten() for v in innovations])

    autocorr = np.zeros(max_lag)
    # Compute autocorrelation for each dimension and average
    for dim in range(m):
        v_dim = V[:, dim]
        v_centered = v_dim - np.mean(v_dim)
        var = np.var(v_dim)
        if var < 1e-15:
            continue
        for lag in range(1, max_lag + 1):
            autocorr[lag - 1] += np.mean(v_centered[lag:] * v_centered[:-lag]) / var
    autocorr /= m

    # Confidence bounds: approximately +/- 1.96 / sqrt(T) for 95% confidence
    from scipy.stats import norm

    bound = norm.ppf(1 - alpha / 2) / np.sqrt(T)

    passed = bool(np.all(np.abs(autocorr) < bound))

    return passed, autocorr
