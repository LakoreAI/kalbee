from typing import List
import numpy as np

from kalbee.modules.utils.linalg import safe_inv
from kalbee.modules.utils.gating import mahalanobis_distance


class JPDAAssociation:
    """
    Joint Probabilistic Data Association (JPDA) helper.

    Computes soft association probabilities beta_{i, j} for associating N tracks
    with M measurements under clutter density clutter_density (lambda) and detection
    probability P_d.
    """

    def __init__(self, p_d: float = 0.9, clutter_density: float = 1e-4, gate_threshold: float = 9.21):
        """
        Args:
            p_d: Probability of detection.
            clutter_density: Spatial density of false alarms (clutter).
            gate_threshold: Mahalanobis distance gating threshold (e.g. 9.21 for 95% 2D gate).
        """
        self.p_d = p_d
        self.clutter_density = clutter_density
        self.gate_threshold = gate_threshold

    def compute_association_probabilities(
        self,
        track_states: List[np.ndarray],
        track_covariances: List[np.ndarray],
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """
        Compute marginal association probabilities matrix beta of shape (N_tracks, M_measurements + 1).

        Column 0 corresponds to no measurement associated with the track (missed detection).
        Columns 1..M correspond to measurement j.

        Args:
            track_states: List of state vectors (n x 1).
            track_covariances: List of covariance matrices (n x n).
            measurement_matrix: Matrix H.
            measurement_covariance: Matrix R.
            measurements: Array of shape (M, m).

        Returns:
            Beta matrix of shape (N_tracks, M + 1) where rows sum to 1.
        """
        N = len(track_states)
        z = np.asarray(measurements, dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1) if z.size > 0 else z.reshape(0, 1)

        M = z.shape[0]

        if N == 0:
            return np.zeros((0, M + 1))

        if M == 0:
            # All tracks have 100% probability of missed detection
            beta = np.zeros((N, 1))
            beta[:, 0] = 1.0
            return beta

        H = measurement_matrix
        R = measurement_covariance
        m = z.shape[1]

        # Compute likelihood matrix L of shape (N, M)
        likelihoods = np.zeros((N, M))
        gated = np.zeros((N, M), dtype=bool)

        for i in range(N):
            x = track_states[i]
            P = track_covariances[i]
            z_pred = H @ x
            S = H @ P @ H.T + R
            S_inv = safe_inv(S)
            det_S = max(1e-10, np.linalg.det(S))

            for j in range(M):
                meas = z[j].reshape(-1, 1)
                innov = meas - z_pred
                d = mahalanobis_distance(innov, S)
                d2 = d ** 2
                if d2 <= self.gate_threshold:
                    gated[i, j] = True
                    exponent = -0.5 * (innov.T @ S_inv @ innov).item()
                    exponent = np.clip(exponent, -700, 700)
                    likelihoods[i, j] = (1.0 / np.sqrt((2 * np.pi) ** m * det_S)) * np.exp(exponent)

        # Compute marginal probabilities (JPDA approximation via normalized likelihoods)
        beta = np.zeros((N, M + 1))

        for i in range(N):
            # Baseline missed detection probability
            beta[i, 0] = 1.0 - self.p_d

            for j in range(M):
                if gated[i, j]:
                    # Likelihood weighted by P_d / clutter_density
                    beta[i, j + 1] = (self.p_d / max(1e-10, self.clutter_density)) * likelihoods[i, j]

            # Normalize across row
            row_sum = np.sum(beta[i])
            if row_sum > 0:
                beta[i] /= row_sum
            else:
                beta[i, 0] = 1.0

        return beta
