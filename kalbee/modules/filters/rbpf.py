from typing import Callable
import numpy as np

from kalbee.modules.filters.base import BaseFilter


class RaoBlackwellizedParticleFilter(BaseFilter):
    """
    Rao-Blackwellized Particle Filter (RBPF).

    Marginalizes out linear states and uses particles only for
    the non-linear portion of the state. This reduces variance
    compared to standard particle filters.

    For a system with state x = [x_linear, x_nonlinear]:
    - Linear states are estimated analytically (like Kalman filter)
    - Non-linear states are estimated via particles

    Usage:
        rbpf = RaoBlackwellizedParticleFilter(
            n_particles=100,
            linear_dim=2,
            nonlinear_dim=2,
            f_linear=f_linear,
            f_nonlinear=f_nonlinear,
            h=h,
            Q_linear=Q_linear,
            Q_nonlinear=Q_nonlinear,
            R=R,
        )
        for z in measurements:
            rbpf.predict(dt=1.0)
            rbpf.update(z)
    """

    def __init__(
        self,
        n_particles: int,
        linear_dim: int,
        nonlinear_dim: int,
        transition_function_linear: Callable[
            [np.ndarray, np.ndarray, float], np.ndarray
        ],
        transition_function_nonlinear: Callable[[np.ndarray, float], np.ndarray],
        measurement_function: Callable[[np.ndarray], np.ndarray],
        process_noise_linear: np.ndarray,
        process_noise_nonlinear: np.ndarray,
        measurement_noise: np.ndarray,
        initial_linear_mean: np.ndarray = None,
        initial_linear_cov: np.ndarray = None,
        initial_nonlinear_mean: np.ndarray = None,
        initial_nonlinear_cov: np.ndarray = None,
    ):
        """
        Initialize the RBPF.

        Args:
            n_particles: Number of particles.
            linear_dim: Dimension of linear sub-state.
            nonlinear_dim: Dimension of non-linear sub-state.
            transition_function_linear: f_linear(x_linear, x_nonlinear, dt) -> x_linear_pred
            transition_function_nonlinear: f_nonlinear(x_nonlinear, dt) -> x_nonlinear_pred
            measurement_function: h(x) -> z (full state)
            process_noise_linear: Q for linear states.
            process_noise_nonlinear: Q for non-linear states.
            measurement_noise: R matrix.
            initial_linear_mean: Initial mean for linear states.
            initial_linear_cov: Initial covariance for linear states.
            initial_nonlinear_mean: Initial mean for non-linear states.
            initial_nonlinear_cov: Initial covariance for non-linear states.
        """
        super().__init__(
            state=np.zeros((linear_dim + nonlinear_dim, 1)),
            covariance=np.eye(linear_dim + nonlinear_dim),
        )

        self.n_particles = n_particles
        self.linear_dim = linear_dim
        self.nonlinear_dim = nonlinear_dim
        self.total_dim = linear_dim + nonlinear_dim

        self.f_linear = transition_function_linear
        self.f_nonlinear = transition_function_nonlinear
        self.h = measurement_function

        self.Q_linear = process_noise_linear
        self.Q_nonlinear = process_noise_nonlinear
        self.R = measurement_noise

        # Initialize particles for non-linear states
        if initial_nonlinear_mean is None:
            initial_nonlinear_mean = np.zeros((nonlinear_dim, 1))
        if initial_nonlinear_cov is None:
            initial_nonlinear_cov = np.eye(nonlinear_dim)

        self.particles_nonlinear = np.random.multivariate_normal(
            initial_nonlinear_mean.flatten(),
            initial_nonlinear_cov,
            size=n_particles,
        )  # (n_particles, nonlinear_dim)

        # Initialize Kalman filters for linear states (one per particle)
        if initial_linear_mean is None:
            initial_linear_mean = np.zeros((linear_dim, 1))
        if initial_linear_cov is None:
            initial_linear_cov = np.eye(linear_dim) * 10.0

        self.linear_means = np.tile(
            initial_linear_mean.flatten(), (n_particles, 1)
        )  # (n_particles, linear_dim)
        self.linear_covs = np.tile(
            initial_linear_cov[np.newaxis, :, :], (n_particles, 1, 1)
        )  # (n_particles, linear_dim, linear_dim)

        # Particle weights
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step: propagate particles and linear states.
        """
        # Propagate non-linear particles
        for i in range(self.n_particles):
            x_nl = self.particles_nonlinear[i].reshape(-1, 1)
            x_nl_pred = self.f_nonlinear(x_nl, dt)
            self.particles_nonlinear[i] = x_nl_pred.flatten()

            # Add process noise
            noise = np.random.multivariate_normal(
                np.zeros(self.nonlinear_dim), self.Q_nonlinear
            )
            self.particles_nonlinear[i] += noise

            # Propagate linear states (conditional on non-linear particle)
            x_l = self.linear_means[i].reshape(-1, 1)
            x_nl = self.particles_nonlinear[i].reshape(-1, 1)
            x_l_pred = self.f_linear(x_l, x_nl, dt)

            # Linear KF predict
            F = self._compute_linear_jacobian(x_l, x_nl, dt)
            self.linear_means[i] = x_l_pred.flatten()
            self.linear_covs[i] = F @ self.linear_covs[i] @ F.T + self.Q_linear

        # Combine for global state estimate (weighted average)
        self._update_global_state()

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step: update weights based on measurement likelihood.
        """
        z = measurement.flatten()

        from kalbee.modules.utils.linalg import safe_inv

        for i in range(self.n_particles):
            # Compute predicted measurement
            x_full = np.concatenate(
                [self.linear_means[i], self.particles_nonlinear[i]]
            ).reshape(-1, 1)
            z_pred = self.h(x_full).flatten()

            # Innovation
            innov = (z - z_pred).reshape(-1, 1)

            # Numerical Jacobian of measurement function w.r.t. linear states
            H_l = self._compute_measurement_linear_jacobian(x_full)

            # Innovation covariance: S = H_l @ P_l @ H_l.T + R
            P_l = self.linear_covs[i]
            S = H_l @ P_l @ H_l.T + self.R

            # Kalman gain for linear state update of particle i
            K = P_l @ H_l.T @ safe_inv(S)

            # Update linear state mean and covariance for particle i
            self.linear_means[i] = (
                self.linear_means[i].reshape(-1, 1) + K @ innov
            ).flatten()
            I_KH = np.eye(self.linear_dim) - K @ H_l
            self.linear_covs[i] = I_KH @ P_l @ I_KH.T + K @ self.R @ K.T
            self.linear_covs[i] = (self.linear_covs[i] + self.linear_covs[i].T) / 2.0

            # Weight update using likelihood
            det_S = max(1e-10, np.linalg.det(S))
            exponent = -0.5 * (innov.T @ safe_inv(S) @ innov).item()
            exponent = np.clip(exponent, -700, 700)
            likelihood = (1.0 / np.sqrt((2 * np.pi) ** len(z) * det_S)) * np.exp(
                exponent
            )
            self.weights[i] *= likelihood

        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Resample if effective sample size is too low
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n_particles / 2:
            self._resample()

        # Update global state
        self._update_global_state()

        return self.state

    def _resample(self):
        """Systematic resampling."""
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights,
        )

        self.particles_nonlinear = self.particles_nonlinear[indices].copy()
        self.linear_means = self.linear_means[indices].copy()
        self.linear_covs = self.linear_covs[indices].copy()
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _update_global_state(self):
        """Compute weighted average of particles for global state."""
        state = np.zeros((self.total_dim, 1))
        for i in range(self.n_particles):
            x_full = np.concatenate(
                [self.linear_means[i], self.particles_nonlinear[i]]
            ).reshape(-1, 1)
            state += self.weights[i] * x_full

        self.state = state

    def _compute_linear_jacobian(self, x_l, x_nl, dt, eps=1e-6):
        """Compute Jacobian of linear transition w.r.t. linear states."""
        n = self.linear_dim
        F = np.eye(n)
        return F

    def _compute_measurement_linear_jacobian(self, x_full, eps=1e-6):
        """Compute numerical Jacobian of measurement function w.r.t. linear states."""
        z0 = self.h(x_full).flatten()
        m = len(z0)
        n_l = self.linear_dim
        H_l = np.zeros((m, n_l))

        for j in range(n_l):
            x_plus = x_full.copy()
            x_plus[j, 0] += eps
            z_plus = self.h(x_plus).flatten()
            H_l[:, j] = (z_plus - z0) / eps

        return H_l
