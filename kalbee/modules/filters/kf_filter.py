from typing import Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class KalmanFilter(BaseFilter):
    """
    Standard Linear Kalman Filter implementation.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        control_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )
        self.control_matrix = control_matrix

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict the next state:
        x = Fx + Bu
        P = FPF' + Q

        Args:
            dt: Time step.
            **kwargs:
                F: Optional override for transition matrix.
                u: Optional control input vector (n_u x 1).
                B: Optional override for control matrix.
        """
        F = self.transition_matrix
        Q = self.transition_covariance

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
        Update the state with a measurement:
        y = z - Hx (residual)
        S = HPH' + R (innovation covariance)
        K = P H' S^-1 (Kalman gain)
        x = x + Ky
        P = (I - KH)P
        """
        z = measurement
        H = self.measurement_matrix
        R = self.measurement_covariance
        P = self.covariance
        x = self.state

        # Innovation
        y = z - H @ x
        # Innovation covariance
        S = H @ P @ H.T + R
        # Kalman gain
        K = P @ H.T @ safe_inv(S)

        # Update state and covariance (Joseph form for numerical stability)
        self.state = x + K @ y
        identity = np.eye(P.shape[0])
        I_KH = identity - K @ H
        self.covariance = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Enforce symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        # Save for IMM and diagnostics
        self.last_y = y
        self.last_S = S

        return self.state
