from typing import List, Tuple, Callable
import numpy as np

from kalbee.modules.utils.linalg import safe_inv
from kalbee.modules.filters.sigma_points import MerweScaledSigmaPoints


class UnscentedRTSSmoother:
    """
    Unscented Rauch-Tung-Striebel (URTS) Smoother.

    A backward smoother for Unscented Kalman Filters (UKF) operating on non-linear systems.

    Usage:
        smoothed_states, smoothed_covariances = UnscentedRTSSmoother.smooth(
            filtered_states,
            filtered_covariances,
            predicted_states,
            predicted_covariances,
            transition_function=f,
            dt=1.0
        )
    """

    @staticmethod
    def smooth(
        filtered_states: List[np.ndarray],
        filtered_covariances: List[np.ndarray],
        predicted_states: List[np.ndarray],
        predicted_covariances: List[np.ndarray],
        transition_function: Callable[[np.ndarray, float], np.ndarray],
        dt: float = 1.0,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run the Unscented RTS backward smoother.

        Args:
            filtered_states: List of filtered state vectors from UKF forward pass.
            filtered_covariances: List of filtered covariance matrices.
            predicted_states: List of predicted state vectors (prior to measurement update).
            predicted_covariances: List of predicted covariance matrices.
            transition_function: System state transition function f(x, dt).
            dt: Time step.
            alpha, beta, kappa: Sigma point scaling parameters.

        Returns:
            Tuple of (smoothed_states, smoothed_covariances).
        """
        N = len(filtered_states)
        if N == 0:
            return [], []

        n = filtered_states[0].shape[0]

        # Initialize sigma points generator
        sigma_pts_gen = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        _, wc = sigma_pts_gen.weights_mean, sigma_pts_gen.weights_cov
        num_sigmas = 2 * n + 1

        smoothed_states = [None] * N
        smoothed_covariances = [None] * N

        # Initialize with last filtered state
        smoothed_states[N - 1] = filtered_states[N - 1].copy()
        smoothed_covariances[N - 1] = filtered_covariances[N - 1].copy()

        for k in range(N - 2, -1, -1):
            x_filtered = filtered_states[k]
            P_filtered = filtered_covariances[k]

            x_pred = predicted_states[k + 1]
            P_pred = predicted_covariances[k + 1]

            # Generate sigma points around filtered state at step k
            sigmas_k = sigma_pts_gen.sigma_points(x_filtered, P_filtered)

            # Propagate through non-linear transition function
            sigmas_k1_pred = np.zeros((num_sigmas, n))
            for i in range(num_sigmas):
                pt = sigmas_k[i].reshape(-1, 1)
                sigmas_k1_pred[i] = transition_function(pt, dt).flatten()

            # Compute cross-covariance P_{k, k+1|k}
            P_cross = np.zeros((n, n))
            for i in range(num_sigmas):
                diff_x = sigmas_k[i].reshape(-1, 1) - x_filtered
                diff_pred = sigmas_k1_pred[i].reshape(-1, 1) - x_pred
                P_cross += wc[i] * (diff_x @ diff_pred.T)

            # Smoother gain: G_k = P_{k, k+1} @ P_{k+1|k}^-1
            G = P_cross @ safe_inv(P_pred)

            # Smoothed state and covariance update
            smoothed_states[k] = x_filtered + G @ (smoothed_states[k + 1] - x_pred)
            smoothed_covariances[k] = (
                P_filtered + G @ (smoothed_covariances[k + 1] - P_pred) @ G.T
            )

            # Enforce symmetry
            smoothed_covariances[k] = (
                smoothed_covariances[k] + smoothed_covariances[k].T
            ) / 2.0

        return smoothed_states, smoothed_covariances
