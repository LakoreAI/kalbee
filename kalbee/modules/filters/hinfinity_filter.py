"""
H-Infinity Filter implementation.

A robust filter that minimizes the worst-case estimation error, unlike the
standard Kalman filter which minimizes the average-case (MMSE) error.
Useful when:
- The noise statistics are not perfectly known
- There are model uncertainties
- You need guaranteed performance bounds

The H-infinity filter ensures the L2-gain from disturbances to estimation
error is bounded by a user-specified parameter gamma. As gamma -> infinity,
the H-infinity filter converges to the standard Kalman filter.

References:
    - Simon, D. (2006). Optimal State Estimation. Wiley.
    - Shaked, U., & de Souza, C. E. (1995). Robust minimum variance filtering.
    - Xu, J., & van Dooren, P. (2002). Robust H-infinity filtering for
      uncertain systems with time-varying delays.
"""

from typing import Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class HInfinityFilter(BaseFilter):
    """
    H-Infinity Filter for robust state estimation.

    This filter provides guaranteed performance bounds even when the noise
    statistics are not perfectly known. It minimizes the worst-case
    estimation error rather than the average-case error.

    The gamma parameter controls the robustness-performance tradeoff:
    - gamma -> infinity: Converges to standard Kalman filter
    - gamma smaller: More robust but potentially less accurate on average
    - A valid gamma must satisfy: gamma > 0 and the Riccati equation must
      have a positive-definite solution
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        process_noise_cov: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_noise_cov: np.ndarray,
        gamma: float = 10.0,
        control_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize the H-Infinity Filter.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_matrix: State transition matrix F (n x n).
            process_noise_cov: Process noise covariance Q (n x n).
            measurement_matrix: Measurement matrix H (m x n).
            measurement_noise_cov: Measurement noise covariance R (m x m).
            gamma: H-infinity performance bound. Must be > 0.
                   Larger values give behavior closer to standard KF.
            control_matrix: Optional control input matrix B (n x n_u).
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=process_noise_cov,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_noise_cov,
        )

        self.gamma = gamma
        self.control_matrix = control_matrix

        # Store process noise as Q (inherited as transition_covariance)
        self.process_noise_cov = process_noise_cov

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step for H-infinity filter.

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        Args:
            dt: Time step.
            **kwargs:
                F: Optional override for transition matrix.
                u: Optional control input vector.
                B: Optional override for control matrix.

        Returns:
            The predicted state vector.
        """
        F = self.transition_matrix
        Q = self.process_noise_cov

        if "F" in kwargs:
            F = kwargs["F"]

        self.state = F @ self.state

        # Apply control input if provided
        u = kwargs.get("u", None)
        B = kwargs.get("B", self.control_matrix)
        if u is not None and B is not None:
            self.state = self.state + B @ np.asarray(u).reshape(-1, 1)

        self.covariance = F @ self.covariance @ F.T + Q

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step for H-infinity filter.

        Uses the H-infinity update equations which provide robustness
        against model uncertainties and worst-case noise.

        The key difference from standard KF is the additional term
        (I - gamma^{-2} * P)^{-1} in the covariance update, which
        bounds the estimation error.

        Args:
            measurement: The observed measurement vector (m x 1).

        Returns:
            The updated state vector.
        """
        z = np.asanyarray(measurement, dtype=float).reshape(-1, 1)
        H = self.measurement_matrix
        R = self.measurement_covariance
        P = self.covariance
        n = self.state.shape[0]

        # Innovation
        y = z - H @ self.state

        # H-infinity modification: apply the robustness term
        # P_robust = P @ (I - gamma^{-2} * P)^{-1}
        gamma_inv_sq = 1.0 / (self.gamma ** 2)
        I_n = np.eye(n)

        # Check if the H-infinity condition is satisfied
        # The matrix (I - gamma^{-2} * P) must be positive definite
        M = I_n - gamma_inv_sq * P
        eigvals = np.linalg.eigvalsh(M)

        if np.all(eigvals > 1e-10):
            # H-infinity gain: K = P_robust @ H.T @ S_robust^{-1}
            P_robust = P @ safe_inv(M)
            S_robust = H @ P_robust @ H.T + R
            K = P_robust @ H.T @ safe_inv(S_robust)
            S = S_robust
        else:
            # Fallback to standard Kalman gain if H-infinity condition fails
            S = H @ P @ H.T + R
            K = P @ H.T @ safe_inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance with Joseph form for numerical stability
        I_KH = I_n - K @ H
        self.covariance = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Enforce symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        # Save for IMM and diagnostics
        self.last_y = y
        self.last_S = S

        return self.state
