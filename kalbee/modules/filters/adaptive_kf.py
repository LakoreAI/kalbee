import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.utils.linalg import safe_inv


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Adaptive Kalman Filter with online estimation of Q and R.

    Uses the innovation-based adaptive estimation (IAE) method to adjust
    process noise (Q) and measurement noise (R) covariances on-the-fly
    based on the innovation sequence.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        window_size: int = 10,
        adapt_Q: bool = True,
        adapt_R: bool = True,
    ):
        """
        Initialize the Adaptive Kalman Filter.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_matrix: State transition matrix F (n x n).
            transition_covariance: Process noise covariance Q (n x n).
            measurement_matrix: Measurement matrix H (m x n).
            measurement_covariance: Measurement noise covariance R (m x m).
            window_size: Number of recent innovations to use for adaptation.
            adapt_Q: Whether to adapt process noise covariance Q.
            adapt_R: Whether to adapt measurement noise covariance R.
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )

        self.window_size = window_size
        self.adapt_Q = adapt_Q
        self.adapt_R = adapt_R

        # Innovation history for adaptive estimation
        self._innovations = []
        self._innovation_covariances = []

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step with adaptive Q/R estimation.

        After the standard KF update, adjusts Q and/or R based on the
        innovation sequence using a sliding window.

        Args:
            measurement: The observed measurement vector (m x 1).

        Returns:
            The updated state vector.
        """
        z = np.asanyarray(measurement).astype(float)
        H = self.measurement_matrix
        R = self.measurement_covariance
        # P here is the *predicted* (prior) covariance, before the KF update.
        # It is the correct P for innovation-based Q/R estimation.
        P_pred = self.covariance
        x = self.state

        # Compute innovation
        y = z - H @ x

        # Innovation covariance (from the predicted covariance)
        S = H @ P_pred @ H.T + R

        # Cache the predicted covariance for the adaptive step, which must use
        # the prior (not the posterior) covariance.
        self._P_pred = P_pred.copy()

        # Store innovation for adaptive estimation
        self._innovations.append(y.copy())
        self._innovation_covariances.append(S.copy())

        # Trim to window size
        if len(self._innovations) > self.window_size:
            self._innovations.pop(0)
            self._innovation_covariances.pop(0)

        # Perform standard KF update
        result = super().update(z, **kwargs)

        # Adaptive estimation after update
        if len(self._innovations) >= 2:
            self._adapt_noise()

        return result

    def _adapt_noise(self):
        """
        Adapt Q and R using the innovation-based approach.

        R_hat = C_v - H @ P_pred @ H.T
        Q_hat = K @ C_v @ K.T

        where C_v = (1/N) * sum(v_k @ v_k.T) is the sample innovation covariance,
        and P_pred is the predicted (prior) covariance from the current step.
        """
        N = len(self._innovations)
        m = self._innovations[0].shape[0]
        H = self.measurement_matrix

        # Predicted (prior) covariance cached during update(); fall back to the
        # current covariance only if adaptation runs before any update.
        P_pred = getattr(self, "_P_pred", self.covariance)

        # Sample innovation covariance
        C_v = np.zeros((m, m))
        for v in self._innovations:
            C_v += v @ v.T
        C_v /= N

        if self.adapt_R:
            # R_hat = C_v - H P_pred H'
            R_hat = C_v - H @ P_pred @ H.T

            # Ensure R stays positive definite
            R_hat = (R_hat + R_hat.T) / 2.0
            eigvals = np.linalg.eigvalsh(R_hat)
            if np.all(eigvals > 0):
                self.measurement_covariance = R_hat

        if self.adapt_Q:
            # Kalman gain for this step, from the predicted covariance
            S = H @ P_pred @ H.T + self.measurement_covariance
            K = P_pred @ H.T @ safe_inv(S)

            # Q_hat = K C_v K'
            Q_hat = K @ C_v @ K.T

            # Ensure Q stays positive semi-definite
            Q_hat = (Q_hat + Q_hat.T) / 2.0
            eigvals = np.linalg.eigvalsh(Q_hat)
            if np.all(eigvals >= 0):
                self.transition_covariance = Q_hat

    def get_innovation_history(self):
        """
        Return the stored innovation sequence (useful for diagnostics).

        Returns:
            List of innovation vectors.
        """
        return list(self._innovations)
