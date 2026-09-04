from typing import List, Tuple
import numpy as np

from kalbee.modules.utils.linalg import safe_inv


class RTSSmoother:
    """
    Rauch-Tung-Striebel (RTS) Smoother.

    A fixed-interval smoother that runs backward over the forward-pass results
    of a Kalman Filter to produce optimally smoothed state estimates.

    Usage:
        1. Run a forward Kalman Filter pass, storing states and covariances.
        2. Call `smooth()` with the stored forward-pass data.
    """

    @staticmethod
    def smooth(
        filtered_states: List[np.ndarray],
        filtered_covariances: List[np.ndarray],
        predicted_states: List[np.ndarray],
        predicted_covariances: List[np.ndarray],
        transition_matrix: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run the RTS backward smoother.

        Args:
            filtered_states: List of filtered (updated) state vectors from forward pass.
            filtered_covariances: List of filtered covariance matrices from forward pass.
            predicted_states: List of predicted state vectors (before update) from forward pass.
            predicted_covariances: List of predicted covariance matrices (before update).
            transition_matrix: State transition matrix F (assumed constant).

        Returns:
            Tuple of (smoothed_states, smoothed_covariances).
        """
        N = len(filtered_states)
        if N == 0:
            return [], []

        F = transition_matrix

        # Initialize with the last filtered state
        smoothed_states = [None] * N
        smoothed_covariances = [None] * N
        smoothed_states[N - 1] = filtered_states[N - 1].copy()
        smoothed_covariances[N - 1] = filtered_covariances[N - 1].copy()

        # Backward pass
        for k in range(N - 2, -1, -1):
            P_filtered = filtered_covariances[k]
            P_predicted = predicted_covariances[k + 1]

            # Smoother gain: G_k = P_k|k @ F^T @ P_{k+1|k}^{-1}
            G = P_filtered @ F.T @ safe_inv(P_predicted)

            # Smoothed state: x_k|N = x_k|k + G_k (x_{k+1|N} - x_{k+1|k})
            smoothed_states[k] = filtered_states[k] + G @ (
                smoothed_states[k + 1] - predicted_states[k + 1]
            )

            # Smoothed covariance: P_k|N = P_k|k + G_k (P_{k+1|N} - P_{k+1|k}) G_k^T
            smoothed_covariances[k] = (
                P_filtered + G @ (smoothed_covariances[k + 1] - P_predicted) @ G.T
            )

            # Enforce symmetry
            smoothed_covariances[k] = (
                smoothed_covariances[k] + smoothed_covariances[k].T
            ) / 2.0

        return smoothed_states, smoothed_covariances
