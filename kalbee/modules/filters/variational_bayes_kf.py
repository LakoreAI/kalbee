import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.utils.linalg import safe_inv


class VariationalBayesKalmanFilter(KalmanFilter):
    """
    Variational Bayesian Adaptive Kalman Filter (VBAKF).

    Estimates both the state and the unknown/time-varying measurement noise
    covariance R online using Variational Inference with Inverse-Wishart priors.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_matrix: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        n_iter: int = 5,
        rho: float = 0.95,
        initial_dof: float = 5.0,
    ):
        """
        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_matrix: Matrix F.
            transition_covariance: Matrix Q.
            measurement_matrix: Matrix H.
            measurement_covariance: Matrix R (prior scale).
            n_iter: Number of Variational Bayes iterations per update.
            rho: Decay factor for prior hyper-parameters (0.9 to 0.99).
            initial_dof: Prior degrees of freedom for Inverse-Wishart.
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
        )

        self.n_iter = n_iter
        self.rho = rho

        m = self.measurement_matrix.shape[0]
        self.dof = initial_dof
        self.scale_matrix = (self.dof - m - 1.0) * self.measurement_covariance.copy()
        if np.any(self.scale_matrix <= 0):
            self.scale_matrix = np.eye(m) * self.dof

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update state and estimate measurement noise covariance R via Variational Bayes.
        """
        z = np.asanyarray(measurement, dtype=float).reshape(-1, 1)
        H = self.measurement_matrix
        m = z.shape[0]
        x_pred = self.state.copy()
        P_pred = self.covariance.copy()

        # Time update for Inverse-Wishart prior hyperparameters
        u_prior = self.rho * (self.dof - m - 1.0) + m + 1.0
        U_prior = self.rho * self.scale_matrix

        u_curr = u_prior
        U_curr = U_prior.copy()

        x_curr = x_pred.copy()
        P_curr = P_pred.copy()

        for _ in range(self.n_iter):
            # Expected inverse measurement noise covariance E[R^-1]
            E_inv_R = u_curr * safe_inv(U_curr)

            # Effective measurement covariance R_eff
            R_eff = safe_inv(E_inv_R)

            # Innovation covariance and Kalman gain
            S = H @ P_pred @ H.T + R_eff
            K = P_pred @ H.T @ safe_inv(S)

            # State and covariance update
            y = z - H @ x_pred
            x_curr = x_pred + K @ y
            I_KH = np.eye(self.state.shape[0]) - K @ H
            P_curr = I_KH @ P_pred @ I_KH.T + K @ R_eff @ K.T

            # Update Inverse-Wishart scale matrix
            innov_post = z - H @ x_curr
            U_curr = U_prior + innov_post @ innov_post.T + H @ P_curr @ H.T
            u_curr = u_prior + 1.0

        self.state = x_curr
        self.covariance = (P_curr + P_curr.T) / 2.0
        self.dof = u_curr
        self.scale_matrix = U_curr
        self.measurement_covariance = U_curr / (u_curr - m - 1.0)

        return self.state
