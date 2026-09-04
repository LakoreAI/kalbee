from typing import Callable
import numpy as np
from math import sqrt

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class CubatureKalmanFilter(BaseFilter):
    """
    Cubature Kalman Filter (CKF) implementation.

    Uses the third-order cubature rule to approximate Gaussian integrals.
    More stable than UKF for high-dimensional systems (n > 5) because
    sigma point counts don't grow exponentially.

    The CKF generates 2n cubature points (not 2n+1 like UKF) using
    symmetric spherical-radial cubature rule.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        transition_function: Callable[[np.ndarray, float], np.ndarray],
        measurement_function: Callable[[np.ndarray], np.ndarray],
    ):
        """
        Initialize the CKF.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_covariance: Process noise covariance (Q).
            measurement_covariance: Measurement noise covariance (R).
            transition_function: f(x, dt) -> predictions.
            measurement_function: h(x) -> measurements.
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_covariance=transition_covariance,
            measurement_covariance=measurement_covariance,
        )

        self.transition_function = transition_function
        self.measurement_function = measurement_function
        self.n = len(state)

    def _cubature_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate 2n cubature points using spherical-radial cubature rule.

        Returns array of shape (2n, n).
        """
        n = self.n
        points = np.zeros((2 * n, n))

        # Cholesky decomposition of P
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Fallback: add small regularization
            S = np.linalg.cholesky(P + np.eye(n) * 1e-10)

        for i in range(n):
            # Positive direction
            points[i] = x.flatten() + sqrt(n) * S[:, i]
            # Negative direction
            points[n + i] = x.flatten() - sqrt(n) * S[:, i]

        return points

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        CKF Predict Step.
        1. Generate 2n cubature points.
        2. Propagate through transition function.
        3. Reconstruct mean and covariance.
        """
        n = self.n

        # 1. Generate cubature points
        points = self._cubature_points(self.state, self.covariance)

        # 2. Propagate through transition function
        pred_points = np.zeros_like(points)
        for i in range(2 * n):
            pt_in = points[i].reshape(-1, 1)
            pt_out = self.transition_function(pt_in, dt)
            pred_points[i] = pt_out.flatten()

        # 3. Reconstruct predicted mean
        self.state = np.mean(pred_points, axis=0).reshape(-1, 1)

        # 4. Reconstruct predicted covariance
        self.covariance = self.transition_covariance.copy()
        for i in range(2 * n):
            diff = pred_points[i].reshape(-1, 1) - self.state
            self.covariance += (diff @ diff.T) / (2 * n)

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        CKF Update Step.
        1. Generate cubature points from predicted state.
        2. Transform through measurement function.
        3. Compute predicted measurement, innovation covariance, cross-covariance.
        4. Compute Kalman gain and update.
        """
        z = measurement
        n = self.n
        m = len(z)

        # 1. Generate cubature points
        points = self._cubature_points(self.state, self.covariance)

        # 2. Transform through measurement function
        meas_points = np.zeros((2 * n, m))
        for i in range(2 * n):
            pt_in = points[i].reshape(-1, 1)
            pt_out = self.measurement_function(pt_in)
            meas_points[i] = pt_out.flatten()

        # 3. Predicted measurement mean
        z_mean = np.mean(meas_points, axis=0).reshape(-1, 1)

        # 4. Innovation covariance
        S = self.measurement_covariance.copy()
        for i in range(2 * n):
            diff = meas_points[i].reshape(-1, 1) - z_mean
            S += (diff @ diff.T) / (2 * n)

        # 5. Cross-covariance
        Pxz = np.zeros((n, m))
        for i in range(2 * n):
            diff_x = points[i].reshape(-1, 1) - self.state
            diff_z = meas_points[i].reshape(-1, 1) - z_mean
            Pxz += (diff_x @ diff_z.T) / (2 * n)

        # 6. Kalman gain
        K = Pxz @ safe_inv(S)

        # 7. Update state and covariance
        y = z - z_mean
        self.state = self.state + K @ y
        self.covariance = self.covariance - K @ S @ K.T

        # Enforce symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        # Save for diagnostics
        self.last_y = y
        self.last_S = S

        return self.state
