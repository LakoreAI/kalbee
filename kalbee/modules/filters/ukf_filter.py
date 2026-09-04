from typing import Callable
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv, safe_cholesky


class UnscentedKalmanFilter(BaseFilter):
    """
    Unscented Kalman Filter (UKF) implementation.

    Uses the Unscented Transform (Sigma Points) to handle non-linearity without
    Jacobians, offering better accuracy than EKF for highly non-linear systems.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        transition_function: Callable[[np.ndarray, float], np.ndarray],
        measurement_function: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Initialize the UKF.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_covariance: Process noise covariance (Q).
            measurement_covariance: Measurement noise covariance (R).
            transition_function: f(x, dt) -> predictions.
            measurement_function: h(x) -> measurements.
            alpha: Spread of sigma points (usually small, e.g. 1e-3).
            beta: Prior knowledge of distribution (2 is optimal for Gaussian).
            kappa: Secondary scaling parameter (usually 0).
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_covariance=transition_covariance,
            measurement_covariance=measurement_covariance,
        )

        self.transition_function = transition_function
        self.measurement_function = measurement_function

        # UT Parameters
        self.n = len(state)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = (alpha**2 * (self.n + kappa)) - self.n

        # Weights initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights used for mean and covariance reconstruction."""
        n_sigma = 2 * self.n + 1
        self.wm = np.zeros(n_sigma)
        self.wc = np.zeros(n_sigma)

        # Scaling factor
        c = self.n + self.lambda_

        self.wm[0] = self.lambda_ / c
        self.wc[0] = self.lambda_ / c + (1 - self.alpha**2 + self.beta)

        for i in range(1, n_sigma):
            self.wm[i] = 1.0 / (2 * c)
            self.wc[i] = 1.0 / (2 * c)

    def _compute_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate Sigma Points.
        Returns array of shape (2n+1, n).
        """
        n = self.n
        sigma_points = np.zeros((2 * n + 1, n))

        # First point: Mean
        sigma_points[0] = x.flatten()

        # Calculate square root of P (upper factor: rows of U are used below)
        c = n + self.lambda_
        S = safe_cholesky((c * P).astype(float), lower=False)

        for i in range(n):
            sigma_points[i + 1] = x.flatten() + S[i]
            sigma_points[n + i + 1] = x.flatten() - S[i]

        return sigma_points

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        UKF Predict Step.
        1. Generate sigma points.
        2. Pass sigma points through transition function f(x).
        3. Reconstruct mean and covariance from transformed points.
        """
        # 1. Generate Sigma Points
        self.sigmas_f = self._compute_sigma_points(self.state, self.covariance)

        # 2. Propagate Sigma Points
        # We assume f takes x (vector) and dt
        sigmas_f_pred = np.zeros_like(self.sigmas_f)
        for i in range(2 * self.n + 1):
            # Pass individual points (as column vectors)
            pt_in = self.sigmas_f[i].reshape(-1, 1)
            pt_out = self.transition_function(pt_in, dt)
            sigmas_f_pred[i] = pt_out.flatten()

        self.sigmas_f = sigmas_f_pred

        # 3. Predict Mean and Covariance
        x_pred = np.dot(self.wm, self.sigmas_f)
        self.state = x_pred.reshape(-1, 1)

        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = self.sigmas_f[i].reshape(-1, 1) - self.state
            P_pred += self.wc[i] * (diff @ diff.T)

        self.covariance = P_pred + self.transition_covariance

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        UKF Update Step.
        1. Resample sigma points from predicted state (optional but recommended).
        2. Transform sigma points through measurement function h(x).
        3. Compute measurement mean and covariance.
        4. Compute Cross-Covariance and Kalman Gain.
        """
        z = measurement

        # Optionally regenerate sigma points from the PREDICTED state
        # (Current implementation reuses propagated sigmas for simplicity,
        # but re-generating is more robust)
        sigmas_pred = self._compute_sigma_points(self.state, self.covariance)

        # 1. Transform through h(x)
        # Assuming measurement space dimension m
        m = len(z)
        sigmas_h = np.zeros((2 * self.n + 1, m))

        for i in range(2 * self.n + 1):
            pt_in = sigmas_pred[i].reshape(-1, 1)
            pt_out = self.measurement_function(pt_in)
            sigmas_h[i] = pt_out.flatten()

        # 2. Predicted Measurement Mean and Covariance
        z_mean = np.dot(self.wm, sigmas_h).reshape(-1, 1)

        S = np.zeros((m, m))
        for i in range(2 * self.n + 1):
            diff = sigmas_h[i].reshape(-1, 1) - z_mean
            S += self.wc[i] * (diff @ diff.T)
        S += self.measurement_covariance

        # 3. Cross Covariance
        Pxz = np.zeros((self.n, m))
        for i in range(2 * self.n + 1):
            diff_x = sigmas_pred[i].reshape(-1, 1) - self.state
            diff_z = sigmas_h[i].reshape(-1, 1) - z_mean
            Pxz += self.wc[i] * (diff_x @ diff_z.T)

        # 4. Kalman Gain
        K = Pxz @ safe_inv(S)

        # 5. Update State and Covariance
        y = z - z_mean  # Residual
        self.state = self.state + K @ y
        self.covariance = self.covariance - K @ S @ K.T

        # Save for IMM and diagnostics
        self.last_y = y
        self.last_S = S

        return self.state
