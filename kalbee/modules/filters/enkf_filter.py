from typing import Callable, Optional, Union
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class EnsembleKalmanFilter(BaseFilter):
    """
    Ensemble Kalman Filter (EnKF) implementation.

    Uses an ensemble of state vectors to represent the probability distribution,
    computing Kalman gain from sample covariances. Suitable for high-dimensional
    systems where maintaining a full covariance matrix is impractical.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        transition_function: Callable[[np.ndarray, float], np.ndarray],
        measurement_function: Callable[[np.ndarray], np.ndarray],
        ensemble_size: int = 50,
        rng: Optional[Union[int, np.random.Generator]] = None,
        vectorized_functions: bool = False,
    ):
        """
        Initialize the Ensemble Kalman Filter.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_covariance: Process noise covariance (Q).
            measurement_covariance: Measurement noise covariance (R).
            transition_function: f(x, dt) -> predicted state.
            measurement_function: h(x) -> expected measurement.
            ensemble_size: Number of ensemble members (N).
            rng: Seed or ``numpy.random.Generator`` for reproducible sampling.
                 If None, a fresh default generator is used.
            vectorized_functions: If True, ``transition_function`` and
                ``measurement_function`` are called once on the whole ensemble
                (shape ``(n, ensemble_size)``) instead of per member. The
                supplied functions must then support column-batched input.
                Defaults to False (per-member).
        """
        super().__init__(
            state=state,
            covariance=covariance,
            transition_covariance=transition_covariance,
            measurement_covariance=measurement_covariance,
        )

        self.transition_function = transition_function
        self.measurement_function = measurement_function
        self.ensemble_size = ensemble_size
        self.rng = np.random.default_rng(rng)
        self.vectorized_functions = vectorized_functions

        n = len(state)
        self.n = n

        # Initialize ensemble from the prior distribution
        mean = state.flatten()
        cov = np.asanyarray(covariance).astype(float)
        self.ensemble = self.rng.multivariate_normal(mean, cov, ensemble_size)
        # Shape: (ensemble_size, n)

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step: propagate each ensemble member through the transition function.

        Args:
            dt: Time step.

        Returns:
            The predicted state (ensemble mean).
        """
        Q = self.transition_covariance

        # Generate process noise perturbations
        process_noise = self.rng.multivariate_normal(
            np.zeros(self.n), Q, self.ensemble_size
        )

        # Propagate ensemble members
        if self.vectorized_functions:
            self.ensemble = (
                self.transition_function(self.ensemble.T, dt).T + process_noise
            )
        else:
            for i in range(self.ensemble_size):
                x_i = self.ensemble[i].reshape(-1, 1)
                x_pred = self.transition_function(x_i, dt)
                self.ensemble[i] = x_pred.flatten() + process_noise[i]

        # Update state and covariance from ensemble
        self._update_statistics()

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step: ensemble-based Kalman update with perturbed observations.

        Args:
            measurement: The observed measurement vector (m x 1).

        Returns:
            The updated state (ensemble mean).
        """
        z = np.asanyarray(measurement).flatten()
        R = self.measurement_covariance
        m = len(z)

        # Transform ensemble through measurement function -> (N, m)
        if self.vectorized_functions:
            ensemble_measurements = np.atleast_2d(
                self.measurement_function(self.ensemble.T)
            ).T
        else:
            ensemble_measurements = np.zeros((self.ensemble_size, m))
            for i in range(self.ensemble_size):
                x_i = self.ensemble[i].reshape(-1, 1)
                z_pred = self.measurement_function(x_i)
                ensemble_measurements[i] = z_pred.flatten()

        # Ensemble means
        x_mean = np.mean(self.ensemble, axis=0)
        z_mean = np.mean(ensemble_measurements, axis=0)

        # Anomaly matrices (deviations from mean)
        A_x = self.ensemble - x_mean  # (N, n)
        A_z = ensemble_measurements - z_mean  # (N, m)

        N = self.ensemble_size

        # Sample cross-covariance P_xz = (1/(N-1)) * A_x^T @ A_z
        P_xz = (A_x.T @ A_z) / (N - 1)  # (n, m)

        # Sample innovation covariance P_zz = (1/(N-1)) * A_z^T @ A_z + R
        P_zz = (A_z.T @ A_z) / (N - 1) + R  # (m, m)

        # Kalman gain
        K = P_xz @ safe_inv(P_zz)  # (n, m)

        # Perturbed observations (stochastic EnKF)
        observation_noise = self.rng.multivariate_normal(np.zeros(m), R, N)

        # Update all ensemble members at once: (N, m) @ (m, n) -> (N, n)
        innovations = (z + observation_noise) - ensemble_measurements  # (N, m)
        self.ensemble = self.ensemble + innovations @ K.T

        # Update state and covariance from ensemble
        self._update_statistics()

        return self.state

    def _update_statistics(self):
        """Recompute state and covariance from the current ensemble."""
        self.state = np.mean(self.ensemble, axis=0).reshape(-1, 1)

        # Sample covariance
        diff = self.ensemble - self.state.flatten()
        self.covariance = (diff.T @ diff) / (self.ensemble_size - 1)

        # Enforce symmetry
        self.covariance = (self.covariance + self.covariance.T) / 2.0
