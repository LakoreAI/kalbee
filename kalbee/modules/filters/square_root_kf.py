import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_cholesky


class SquareRootKalmanFilter(BaseFilter):
    """
    Square-Root Kalman Filter (SRKF) implementation.

    Maintains the Cholesky factor of the covariance matrix instead of the covariance
    matrix itself. This improves numerical stability and prevents the covariance
    from losing positive-definiteness due to rounding errors.
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
        # super().__init__ will trigger the covariance setter which sets self.S_P
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )
        # Store Cholesky factors for process and measurement noise.
        # safe_cholesky adds small diagonal regularization if the matrices are
        # positive semi-definite (e.g. rank-deficient Q).
        self.S_Q = safe_cholesky(self.transition_covariance)
        self.S_R = safe_cholesky(self.measurement_covariance)

        self._stabilize_cholesky()

    def _stabilize_cholesky(self):
        """Enforce lower triangular structure and positive diagonal elements for S_P."""
        self.S_P = np.tril(self.S_P)
        for i in range(self.S_P.shape[1]):
            diag_val = self.S_P[i, i]
            if diag_val < 0:
                self.S_P[:, i] = -self.S_P[:, i]
            elif diag_val == 0:
                self.S_P[i, i] = 1e-9

    @property
    def covariance(self) -> np.ndarray:
        """Reconstruct covariance matrix from Cholesky factor."""
        return self.S_P @ self.S_P.T

    @covariance.setter
    def covariance(self, cov: np.ndarray):
        """Update Cholesky factor when covariance is updated."""
        self.S_P = safe_cholesky(cov)
        self._stabilize_cholesky()

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step using QR decomposition of the transition composite matrix.
        """
        F = self.transition_matrix
        if "F" in kwargs:
            F = kwargs["F"]

        # Predict state: x = Fx
        self.state = F @ self.state

        # Predict Cholesky factor using QR decomposition
        # Form M_T = [ (F S_P)^T ]
        #            [   S_Q^T   ]
        M_T = np.vstack(((F @ self.S_P).T, self.S_Q.T))
        _, R_qr = np.linalg.qr(M_T)
        self.S_P = R_qr.T

        self._stabilize_cholesky()
        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step using QR decomposition of the measurement composite matrix.
        """
        z = measurement
        H = self.measurement_matrix

        m = z.shape[0]
        n = self.state.shape[0]

        # Form composite matrix A_T = [ S_R^T        0    ]
        #                             [ (H S_P)^T   S_P^T ]
        A_T = np.zeros((m + n, m + n))
        A_T[:m, :m] = self.S_R.T
        A_T[m:, :m] = (H @ self.S_P).T
        A_T[m:, m:] = self.S_P.T

        _, R_qr = np.linalg.qr(A_T)

        # Partition R_qr = [ X  Y ]
        #                  [ 0  Z ]
        # where X is (m x m), Y is (m x n), Z is (n x n)
        X = R_qr[:m, :m]
        Y = R_qr[:m, m:]
        Z = R_qr[m:, m:]

        # Solve X @ K.T = Y for K (X is upper-triangular from QR)
        K = np.linalg.solve(X, Y).T

        # Update state: x = x + K(z - Hx)
        y = z - H @ self.state
        self.state = self.state + K @ y

        # Updated Cholesky factor: S_P = Z.T
        self.S_P = Z.T

        # Save for IMM and diagnostics
        self.last_y = y
        self.last_S = X.T @ X

        self._stabilize_cholesky()
        return self.state
