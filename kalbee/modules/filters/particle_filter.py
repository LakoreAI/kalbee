from typing import Callable, Optional, Union
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class ParticleFilter(BaseFilter):
    """
    Particle Filter (Sequential Monte Carlo) implementation.

    Handles non-linear, non-Gaussian systems using a set of weighted particles.
    Uses systematic resampling to prevent particle degeneracy.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transition_function: Callable[[np.ndarray, float], np.ndarray],
        measurement_function: Callable[[np.ndarray], np.ndarray],
        measurement_covariance: np.ndarray,
        noise_function: Optional[Callable[[int], np.ndarray]] = None,
        num_particles: int = 500,
        resample_threshold: float = 0.5,
        process_noise_cov: Optional[np.ndarray] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
        vectorized_functions: bool = False,
    ):
        """
        Initialize the Particle Filter.

        Args:
            state: Initial state vector (n x 1).
            covariance: Initial state covariance matrix (n x n).
            transition_function: f(x, dt) -> predicted state for a single particle.
            measurement_function: h(x) -> expected measurement for a single particle.
            measurement_covariance: Measurement noise covariance (R).
            noise_function: Optional function(num_particles) -> noise matrix (N x n).
                            If None, samples zero-mean Gaussian process noise with
                            covariance ``process_noise_cov``.
            num_particles: Number of particles.
            resample_threshold: Fraction of N_eff/N below which resampling is triggered.
            process_noise_cov: Process noise covariance used when ``noise_function``
                               is None. Defaults to ``0.1 * I``.
            rng: Seed or ``numpy.random.Generator`` for reproducible sampling.
                 If None, a fresh default generator is used.
            vectorized_functions: If True, ``transition_function`` and
                ``measurement_function`` are called once on the whole particle
                batch (shape ``(n, num_particles)``) instead of per particle,
                which is dramatically faster. The supplied functions must then
                support column-batched input. Defaults to False (per-particle).
        """
        super().__init__(
            state=state,
            covariance=covariance,
            measurement_covariance=measurement_covariance,
        )

        self.transition_function = transition_function
        self.measurement_function = measurement_function
        self.noise_function = noise_function
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.rng = np.random.default_rng(rng)
        self.vectorized_functions = vectorized_functions

        n = len(state)
        self.n = n

        self.process_noise_cov = (
            np.eye(n) * 0.1
            if process_noise_cov is None
            else np.asanyarray(process_noise_cov).astype(float)
        )

        # Initialize particles from the prior distribution
        mean = state.flatten()
        cov = np.asanyarray(covariance).astype(float)
        self.particles = self.rng.multivariate_normal(mean, cov, num_particles)
        # Shape: (num_particles, n)

        # Uniform weights initially
        self.weights = np.ones(num_particles) / num_particles

    def _systematic_resample(self):
        """
        Systematic resampling to prevent particle degeneracy.
        Replaces low-weight particles with copies of high-weight ones.
        """
        N = self.num_particles
        positions = (np.arange(N) + self.rng.uniform()) / N
        cumulative_weights = np.cumsum(self.weights)

        indices = np.searchsorted(cumulative_weights, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(N) / N

    def _effective_particles(self) -> float:
        """Return the effective number of particles (N_eff)."""
        return 1.0 / np.sum(self.weights**2)

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step: propagate each particle through the transition function and add noise.

        Args:
            dt: Time step.

        Returns:
            The predicted state (weighted mean of particles).
        """
        # Propagate particles
        if self.vectorized_functions:
            # Single batched call: (n, N) -> (n, N)
            self.particles = self.transition_function(self.particles.T, dt).T
        else:
            for i in range(self.num_particles):
                x_i = self.particles[i].reshape(-1, 1)
                x_pred = self.transition_function(x_i, dt)
                self.particles[i] = x_pred.flatten()

        # Add process noise
        if self.noise_function is not None:
            noise = self.noise_function(self.num_particles)
        else:
            noise = self.rng.multivariate_normal(
                np.zeros(self.n), self.process_noise_cov, self.num_particles
            )
        self.particles += noise

        # Update state estimate
        self.state = np.average(self.particles, weights=self.weights, axis=0).reshape(
            -1, 1
        )

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step: compute particle weights based on measurement likelihood and resample.

        Args:
            measurement: The observed measurement vector (m x 1).

        Returns:
            The updated state (weighted mean of particles).
        """
        z = np.asanyarray(measurement).flatten()
        R = self.measurement_covariance

        # Compute weights based on measurement likelihood (Gaussian)
        R_inv = safe_inv(R)
        R_det = np.linalg.det(R)
        m = len(z)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** m * R_det)

        # Predicted measurements for all particles -> (N, m)
        if self.vectorized_functions:
            z_pred_all = np.atleast_2d(self.measurement_function(self.particles.T)).T
        else:
            z_pred_all = np.empty((self.num_particles, m))
            for i in range(self.num_particles):
                x_i = self.particles[i].reshape(-1, 1)
                z_pred_all[i] = self.measurement_function(x_i).flatten()

        # Vectorized Gaussian likelihood over all particles
        diff = z - z_pred_all  # (N, m)
        exponents = -0.5 * np.einsum("ij,jk,ik->i", diff, R_inv, diff)
        self.weights *= norm_const * np.exp(exponents)

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum < 1e-300:
            # All weights are essentially zero — reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= weight_sum

        # Resample if effective particle count is too low
        n_eff = self._effective_particles()
        if n_eff < self.resample_threshold * self.num_particles:
            self._systematic_resample()

        # Update state and covariance estimates
        self.state = np.average(self.particles, weights=self.weights, axis=0).reshape(
            -1, 1
        )

        # Weighted covariance (vectorized): sum_i w_i (x_i - mu)(x_i - mu)^T
        diff_matrix = self.particles - self.state.flatten()  # (N, n)
        self.covariance = (self.weights[:, None] * diff_matrix).T @ diff_matrix

        return self.state
