from typing import Tuple, Optional
import numpy as np


class OnlineEM:
    """
    Online Expectation-Maximization for Kalman filter parameters.

    Continuously estimates Q and R from streaming measurements
    without storing the full history. Uses a forgetting factor
    to weight recent data more heavily.

    Usage:
        em = OnlineEM(F, H, forgetting_factor=0.99)
        for z in measurements:
            em.update(z)
            Q, R = em.get_parameters()
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        measurement_matrix: np.ndarray,
        forgetting_factor: float = 0.99,
        initial_Q: Optional[np.ndarray] = None,
        initial_R: Optional[np.ndarray] = None,
        min_samples: int = 10,
    ):
        """
        Initialize Online EM.

        Args:
            transition_matrix: F matrix (constant).
            measurement_matrix: H matrix (constant).
            forgetting_factor: Exponential decay factor (0-1). Closer to 1 = more memory.
            initial_Q: Initial process noise estimate.
            initial_R: Initial measurement noise estimate.
            min_samples: Minimum samples before updating estimates.
        """
        self.F = transition_matrix
        self.H = measurement_matrix
        self.gamma = forgetting_factor
        self.min_samples = min_samples

        self.n = transition_matrix.shape[0]
        self.m = measurement_matrix.shape[0]

        # Initialize parameters
        self.Q = initial_Q if initial_Q is not None else np.eye(self.n) * 0.01
        self.R = initial_R if initial_R is not None else np.eye(self.m) * 0.1

        # Sufficient statistics (exponential moving averages)
        self._count = 0
        self._E_xx = np.zeros((self.n, self.n))
        self._E_xz = np.zeros((self.n, self.m))
        self._E_zz = np.zeros((self.m, self.m))
        self._E_xprev_x = np.zeros((self.n, self.n))

    def update(
        self,
        filtered_state: np.ndarray,
        filtered_covariance: np.ndarray,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update parameter estimates with new data.

        Args:
            filtered_state: x_k|k (n x 1).
            filtered_covariance: P_k|k (n x n).
            predicted_state: x_k|k-1 (n x 1).
            predicted_covariance: P_k|k-1 (n x n).
            measurement: z_k (m x 1).

        Returns:
            Tuple of (Q, R) updated estimates.
        """
        self._count += 1

        # Compute sufficient statistics
        x = filtered_state.flatten()
        x_pred = predicted_state.flatten()
        z = measurement.flatten()

        # E-step: compute expectations
        # Update sufficient statistics with forgetting factor
        self._E_xx = self.gamma * self._E_xx + np.outer(x, x)
        self._E_xz = self.gamma * self._E_xz + np.outer(x, z)
        self._E_zz = self.gamma * self._E_zz + np.outer(z, z)
        self._E_xprev_x = self.gamma * self._E_xprev_x + np.outer(x_pred, x)

        # M-step: update Q and R
        if self._count >= self.min_samples:
            # Effective sample count
            eff_count = (1 - self.gamma ** self._count) / (1 - self.gamma)

            # Update Q
            A = self._E_xx / eff_count
            B = self._E_xprev_x / eff_count
            self.Q = A - self.F @ B
            self.Q = (self.Q + self.Q.T) / 2.0

            # Ensure positive definite
            min_diag = 1e-8
            self.Q = np.maximum(self.Q, np.diag(np.diag(self.Q)))
            self.Q = np.maximum(self.Q, np.eye(self.n) * min_diag)

            # Update R
            C = self._E_zz / eff_count
            D = self._E_xz / eff_count
            self.R = C - self.H @ D
            self.R = (self.R + self.R.T) / 2.0

            # Ensure positive definite
            self.R = np.maximum(self.R, np.diag(np.diag(self.R)))
            self.R = np.maximum(self.R, np.eye(self.m) * min_diag)

        return self.Q.copy(), self.R.copy()

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current Q and R estimates."""
        return self.Q.copy(), self.R.copy()

    @property
    def sample_count(self) -> int:
        """Number of samples processed."""
        return self._count
