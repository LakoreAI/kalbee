import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv as _safe_inv


class InformationFilter(BaseFilter):
    """
    Information Filter implementation — the dual of the Kalman Filter.

    Operates in information space using the information matrix (Y = P^{-1})
    and information state vector (y_hat = P^{-1} x). This form is particularly
    useful for multi-sensor fusion, where incorporating new measurements is
    simply an additive operation.
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
        Initialize the Information Filter.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_matrix: State transition matrix F (n x n).
            transition_covariance: Process noise covariance Q (n x n).
            measurement_matrix: Measurement matrix H (m x n).
            measurement_covariance: Measurement noise covariance R (m x m).
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )

        # Convert to information space
        P = np.asanyarray(covariance).astype(float)
        x = np.asanyarray(state).astype(float)

        self.info_matrix = _safe_inv(P)  # Y = P^{-1}
        self.info_state = self.info_matrix @ x  # y_hat = P^{-1} x

    @property
    def Y(self) -> np.ndarray:
        """Information matrix (inverse of covariance)."""
        return self.info_matrix

    @property
    def y_hat(self) -> np.ndarray:
        """Information state vector (Y @ x)."""
        return self.info_state

    def _sync_state_space(self):
        """Convert information space back to state space for external access."""
        self.covariance = _safe_inv(self.info_matrix)
        self.state = self.covariance @ self.info_state

        # Enforce symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2.0

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step in information space.

        Y_pred = (F^{-T} Y F^{-1} + Q^{-1})  -- simplified via M approach
        Actually uses: M = F^{-T} Y F^{-1}, then
        Y_pred = M - M(M + Q^{-1})^{-1} M  (matrix inversion lemma)

        For numerical stability, we convert to state space, predict, then convert back.

        Args:
            dt: Time step.

        Returns:
            The predicted state vector.
        """
        F = self.transition_matrix
        Q = self.transition_covariance

        if "F" in kwargs:
            F = kwargs["F"]

        # Convert to state space for prediction (more numerically stable)
        P = _safe_inv(self.info_matrix)
        x = P @ self.info_state

        # Standard KF predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Convert back to information space
        self.info_matrix = _safe_inv(P_pred)
        self.info_state = self.info_matrix @ x_pred

        # Sync state space
        self._sync_state_space()

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step in information space — simple additive update.

        Y_k = Y_{k|k-1} + H^T R^{-1} H
        y_hat_k = y_hat_{k|k-1} + H^T R^{-1} z

        Args:
            measurement: The observed measurement vector (m x 1).

        Returns:
            The updated state vector.
        """
        z = np.asanyarray(measurement).astype(float)
        H = self.measurement_matrix
        R = self.measurement_covariance

        R_inv = _safe_inv(R)

        # Information update (additive!)
        self.info_matrix += H.T @ R_inv @ H
        self.info_state += H.T @ R_inv @ z

        # Sync state space
        self._sync_state_space()

        return self.state

    def fuse(self, info_matrix_ext: np.ndarray, info_state_ext: np.ndarray):
        """
        Fuse information from an external source (e.g., another sensor's information filter).

        This is the key advantage of the information filter: fusion is simply addition.

        Args:
            info_matrix_ext: External information matrix contribution.
            info_state_ext: External information state contribution.
        """
        self.info_matrix += info_matrix_ext
        self.info_state += info_state_ext
        self._sync_state_space()
