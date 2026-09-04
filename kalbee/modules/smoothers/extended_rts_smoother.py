from typing import List, Tuple
import numpy as np

from kalbee.modules.utils.linalg import safe_inv


class ExtendedRTSSmoother:
    """
    Extended Rauch-Tung-Striebel (RTS) Smoother for EKF.

    Works with non-linear systems by using linearized Jacobians.
    The smoother runs backward over the EKF forward pass results to
    produce optimally smoothed state estimates.

    Usage:
        1. Run a forward EKF pass, storing states, covariances, and transition functions.
        2. Call `smooth()` with the stored forward-pass data.
    """

    @staticmethod
    def smooth(
        filtered_states: List[np.ndarray],
        filtered_covariances: List[np.ndarray],
        predicted_states: List[np.ndarray],
        predicted_covariances: List[np.ndarray],
        transition_function: callable = None,
        state_dim: int = None,
        dt: float = 1.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run the Extended RTS backward smoother.

        Args:
            filtered_states: List of filtered (updated) state vectors from EKF.
            filtered_covariances: List of filtered covariance matrices from EKF.
            predicted_states: List of predicted state vectors from EKF.
            predicted_covariances: List of predicted covariance matrices from EKF.
            transition_function: Non-linear transition function f(x, dt) for Jacobian computation.
            state_dim: State dimension (required if transition_function is None).
            dt: Time step for Jacobian computation.

        Returns:
            Tuple of (smoothed_states, smoothed_covariances).
        """
        N = len(filtered_states)
        if N == 0:
            return [], []

        n = state_dim or len(filtered_states[0])

        # Initialize with the last filtered state
        smoothed_states = [None] * N
        smoothed_covariances = [None] * N
        smoothed_states[N - 1] = filtered_states[N - 1].copy()
        smoothed_covariances[N - 1] = filtered_covariances[N - 1].copy()

        # Backward pass
        for k in range(N - 2, -1, -1):
            P_filtered = filtered_covariances[k]
            P_predicted = predicted_covariances[k + 1]

            # Compute Jacobian of transition function
            if transition_function is not None:
                F = ExtendedRTSSmoother._compute_jacobian(
                    filtered_states[k], transition_function, dt, n
                )
            else:
                # Fall back to linear RTS smoother if no transition function
                from kalbee.modules.smoothers.rts_smoother import RTSSmoother
                return RTSSmoother.smooth(
                    filtered_states, filtered_covariances,
                    predicted_states, predicted_covariances,
                    transition_matrix=np.eye(n)
                )

            # Smoother gain: G_k = P_k|k @ F^T @ P_{k+1|k}^{-1}
            G = P_filtered @ F.T @ safe_inv(P_predicted)

            # Smoothed state
            smoothed_states[k] = filtered_states[k] + G @ (
                smoothed_states[k + 1] - predicted_states[k + 1]
            )

            # Smoothed covariance
            smoothed_covariances[k] = (
                P_filtered + G @ (smoothed_covariances[k + 1] - P_predicted) @ G.T
            )

            # Enforce symmetry
            smoothed_covariances[k] = (
                smoothed_covariances[k] + smoothed_covariances[k].T
            ) / 2.0

        return smoothed_states, smoothed_covariances

    @staticmethod
    def _compute_jacobian(
        state: np.ndarray,
        transition_function: callable,
        dt: float,
        n: int,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute Jacobian of transition function numerically.

        F_ij = df_i/dx_j

        Args:
            state: Current state vector (n x 1).
            transition_function: f(x, dt) -> x_pred.
            dt: Time step.
            n: State dimension.
            eps: Perturbation for finite differences.

        Returns:
            Jacobian matrix F (n x n).
        """
        F = np.zeros((n, n))

        for j in range(n):
            x_plus = state.copy()
            x_plus[j, 0] += eps
            f_plus = transition_function(x_plus, dt).flatten()

            x_minus = state.copy()
            x_minus[j, 0] -= eps
            f_minus = transition_function(x_minus, dt).flatten()

            F[:, j] = (f_plus - f_minus) / (2 * eps)

        return F
