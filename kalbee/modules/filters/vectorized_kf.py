import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import batched_inv


class VectorizedKalmanFilter(BaseFilter):
    """
    Vectorized Kalman Filter (VKF) implementation.

    Optimized for tracking a large batch of independent targets simultaneously.
    Expects state tensors of shape (batch_size, n, 1) and covariance tensors of
    shape (batch_size, n, n).
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ):
        """
        Initialize the Vectorized Kalman Filter.

        Args:
            state: Initial state tensor of shape (batch_size, n, 1)
            covariance: Initial state covariance of shape (batch_size, n, n)
            transition_matrix: State transition matrix F of shape (n, n) or (batch_size, n, n)
            transition_covariance: Process noise covariance Q of shape (n, n) or (batch_size, n, n)
            measurement_matrix: Measurement matrix H of shape (m, n) or (batch_size, m, n)
            measurement_covariance: Measurement noise covariance R of shape (m, m) or (batch_size, m, m)
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )
        # Ensure input dimensions are formatted as (batch_size, dim, 1 or dim)
        self.state = np.asanyarray(state).astype(float)
        self.covariance = np.asanyarray(covariance).astype(float)

        self.batch_size = self.state.shape[0]
        self.n = self.state.shape[1]

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step in batch:
        x = F @ x
        P = F @ P @ F.T + Q
        """
        F = self.transition_matrix
        if "F" in kwargs:
            F = kwargs["F"]
        Q = self.transition_covariance

        # Perform batched state prediction: (B, n, n) @ (B, n, 1) -> (B, n, 1)
        # NumPy's @ operator automatically handles batch dimensions!
        self.state = F @ self.state

        # Perform batched covariance prediction: F @ P @ F.T + Q
        # If F has shape (n, n) or (B, n, n), F.T for batch is np.swapaxes(F, -1, -2)
        F_T = np.swapaxes(F, -1, -2) if F.ndim == 3 else F.T
        self.covariance = F @ self.covariance @ F_T + Q

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step in batch:
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ S^-1
        x = x + K @ y
        P = (I - K @ H) @ P
        """
        z = np.asanyarray(measurement).astype(float)
        H = self.measurement_matrix
        R = self.measurement_covariance
        P = self.covariance
        x = self.state

        H_T = np.swapaxes(H, -1, -2) if H.ndim == 3 else H.T

        # Batched residual
        y = z - H @ x

        # Batched innovation covariance
        S = H @ P @ H_T + R

        # Batched matrix inversion (with regularized fallback)
        S_inv = batched_inv(S)

        # Batched Kalman gain
        K = P @ H_T @ S_inv

        # Batched state update
        self.state = x + K @ y

        # Batched covariance update (Joseph form for stability)
        identity = np.eye(self.n)
        # Expand identity to batch shape (B, n, n)
        I_batch = np.repeat(identity[np.newaxis, :, :], self.batch_size, axis=0)

        I_KH = I_batch - K @ H
        I_KH_T = np.swapaxes(I_KH, -1, -2)
        K_T = np.swapaxes(K, -1, -2)

        self.covariance = I_KH @ P @ I_KH_T + K @ R @ K_T

        # Enforce symmetry
        self.covariance = (self.covariance + np.swapaxes(self.covariance, -1, -2)) / 2.0

        return self.state
