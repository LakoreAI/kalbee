"""
UKF with pluggable sigma point strategies.

A flexible UKF implementation that accepts different sigma point generation
strategies via the Strategy pattern. This allows users to easily switch between
different sigma point formulations without changing the filter logic.

References:
    - Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation.
    - Van der Merwe, R., & Wan, E. A. (2001). The unscented Kalman filter for nonlinear estimation.
"""

from typing import Callable, Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.filters.sigma_points import SigmaPoints, SimplexSigmaPoints
from kalbee.modules.utils.linalg import safe_inv


class SigmaPointUKF(BaseFilter):
    """
    Unscented Kalman Filter with pluggable sigma point strategy.

    This UKF variant allows you to inject any sigma point generation strategy
    (SimplexSigmaPoints, MerweScaledSigmaPoints, JulierSigmaPoints, or custom).
    This provides maximum flexibility for tuning the filter to specific problems.

    Unlike the standard UKF which computes sigma points internally, this class
    delegates sigma point generation to an external strategy object.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        transition_function: Callable[[np.ndarray, float], np.ndarray],
        measurement_function: Callable[[np.ndarray], np.ndarray],
        sigma_points: Optional[SigmaPoints] = None,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Initialize the SigmaPointUKF.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_covariance: Process noise covariance (Q).
            measurement_covariance: Measurement noise covariance (R).
            transition_function: f(x, dt) -> predicted state.
            measurement_function: h(x) -> expected measurement.
            sigma_points: Sigma point strategy object. If None, uses
                         SimplexSigmaPoints with the given alpha/beta/kappa.
            alpha: Spread of sigma points (used if sigma_points is None).
            beta: Prior knowledge (used if sigma_points is None).
            kappa: Secondary scaling (used if sigma_points is None).
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_covariance=transition_covariance,
            measurement_covariance=measurement_covariance,
        )

        self.transition_function = transition_function
        self.measurement_function = measurement_function

        # Use provided sigma points or create default
        if sigma_points is None:
            self.sigma_points = SimplexSigmaPoints(
                n=len(state), alpha=alpha, beta=beta, kappa=kappa
            )
        else:
            self.sigma_points = sigma_points

        self.n = len(state)

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        UKF Predict Step with pluggable sigma points.

        1. Generate sigma points using the strategy.
        2. Propagate sigma points through transition function.
        3. Reconstruct mean and covariance from transformed points.

        Args:
            dt: Time step.
            **kwargs: Not used, kept for interface compatibility.

        Returns:
            The predicted state vector.
        """
        # 1. Generate sigma points
        sigmas = self.sigma_points.sigma_points(self.state, self.covariance)
        wm = self.sigma_points.weights_mean
        wc = self.sigma_points.weights_cov

        # 2. Propagate sigma points
        n_sigma = sigmas.shape[0]
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(n_sigma):
            pt_in = sigmas[i].reshape(-1, 1)
            pt_out = self.transition_function(pt_in, dt)
            sigmas_pred[i] = pt_out.flatten()

        # 3. Predict mean and covariance
        x_pred = np.dot(wm, sigmas_pred)
        self.state = x_pred.reshape(-1, 1)

        P_pred = np.zeros((self.n, self.n))
        for i in range(n_sigma):
            diff = sigmas_pred[i].reshape(-1, 1) - self.state
            P_pred += wc[i] * (diff @ diff.T)

        self.covariance = P_pred + self.transition_covariance

        # Store sigma points for update step
        self._sigmas_pred = sigmas_pred
        self._wm = wm
        self._wc = wc

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        UKF Update Step with pluggable sigma points.

        1. Transform sigma points through measurement function.
        2. Compute measurement mean and covariance.
        3. Compute cross-covariance and Kalman gain.
        4. Update state and covariance.

        Args:
            measurement: The observed measurement vector (m x 1).

        Returns:
            The updated state vector.
        """
        z = np.asanyarray(measurement, dtype=float).reshape(-1, 1)
        m = z.shape[0]
        wm = self._wm
        wc = self._wc
        sigmas_pred = self._sigmas_pred

        # 1. Transform through h(x)
        sigmas_h = np.zeros((sigmas_pred.shape[0], m))
        for i in range(sigmas_pred.shape[0]):
            pt_in = sigmas_pred[i].reshape(-1, 1)
            pt_out = self.measurement_function(pt_in)
            sigmas_h[i] = pt_out.flatten()

        # 2. Predicted measurement mean and covariance
        z_mean = np.dot(wm, sigmas_h).reshape(-1, 1)

        S = np.zeros((m, m))
        for i in range(sigmas_h.shape[0]):
            diff = sigmas_h[i].reshape(-1, 1) - z_mean
            S += wc[i] * (diff @ diff.T)
        S += self.measurement_covariance

        # 3. Cross covariance
        Pxz = np.zeros((self.n, m))
        for i in range(sigmas_h.shape[0]):
            diff_x = sigmas_pred[i].reshape(-1, 1) - self.state
            diff_z = sigmas_h[i].reshape(-1, 1) - z_mean
            Pxz += wc[i] * (diff_x @ diff_z.T)

        # 4. Kalman gain
        K = Pxz @ safe_inv(S)

        # 5. Update state and covariance
        y = z - z_mean
        self.state = self.state + K @ y
        self.covariance = self.covariance - K @ S @ K.T

        # Save for IMM and diagnostics
        self.last_y = y
        self.last_S = S

        return self.state
