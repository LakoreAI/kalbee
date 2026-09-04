from typing import List
import numpy as np

from kalbee.modules.utils.linalg import safe_inv
from kalbee.modules.utils.gating import mahalanobis_distance


class BernoulliTarget:
    """Represents a single detected target hypothesis in PMBM."""

    def __init__(
        self, state: np.ndarray, covariance: np.ndarray, existence_prob: float = 0.5
    ):
        self.state = state.copy()
        self.covariance = covariance.copy()
        self.existence_prob = existence_prob
        self.id = id(self)


class PMBMTracker:
    """
    Poisson Multi-Bernoulli Mixture (PMBM) Filter for Multi-Target Tracking.

    Tracks both detected targets (Multi-Bernoulli mixture) and undetected targets
    (Poisson point process) within a Random Finite Set (RFS) framework.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        process_covariance: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        p_d: float = 0.9,
        p_s: float = 0.99,
        clutter_density: float = 1e-4,
        birth_rate: float = 0.05,
        prune_threshold: float = 1e-3,
    ):
        """
        Args:
            transition_matrix: Matrix F.
            process_covariance: Matrix Q.
            measurement_matrix: Matrix H.
            measurement_covariance: Matrix R.
            p_d: Probability of detection.
            p_s: Probability of target survival.
            clutter_density: Spatial clutter density lambda_c.
            birth_rate: Expected rate of new target births.
            prune_threshold: Minimum existence probability before pruning.
        """
        self.F = transition_matrix
        self.Q = process_covariance
        self.H = measurement_matrix
        self.R = measurement_covariance
        self.p_d = p_d
        self.p_s = p_s
        self.clutter_density = clutter_density
        self.birth_rate = birth_rate
        self.prune_threshold = prune_threshold

        self.targets: List[BernoulliTarget] = []

    def predict(self, dt: float = 1.0):
        """Predict step for Poisson and Bernoulli target components."""
        for target in self.targets:
            # Predict state and covariance
            target.state = self.F @ target.state
            target.covariance = self.F @ target.covariance @ self.F.T + self.Q
            # Predict survival probability
            target.existence_prob *= self.p_s

    def update(self, measurements: np.ndarray) -> List[BernoulliTarget]:
        """
        Update step with new measurement frame.

        Args:
            measurements: Array of shape (M, m) or (M, m, 1).

        Returns:
            List of confirmed target hypotheses (r > 0.5).
        """
        z = np.asarray(measurements, dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1) if z.size > 0 else z.reshape(0, 1)

        M = z.shape[0]
        H = self.H
        R = self.R

        updated_targets = []

        # 1. Update existing Bernoulli targets
        for target in self.targets:
            x = target.state
            P = target.covariance
            r = target.existence_prob

            # Missed detection hypothesis for target
            r_missed = r * (1.0 - self.p_d) / (1.0 - r + r * (1.0 - self.p_d))
            updated_targets.append(BernoulliTarget(x, P, existence_prob=r_missed))

            # Detection hypotheses
            if M > 0:
                S = H @ P @ H.T + R
                S_inv = safe_inv(S)
                z_pred = H @ x

                for j in range(M):
                    meas = z[j].reshape(-1, 1)
                    d2 = mahalanobis_distance(meas - z_pred, S) ** 2
                    if d2 <= 16.0:  # Gating
                        innov = meas - z_pred
                        K = P @ H.T @ S_inv
                        x_upd = x + K @ innov
                        P_upd = (np.eye(P.shape[0]) - K @ H) @ P

                        det_S = max(1e-10, np.linalg.det(S))
                        likelihood = (
                            1.0 / np.sqrt((2 * np.pi) ** meas.shape[0] * det_S)
                        ) * np.exp(-0.5 * d2)
                        r_det = (r * self.p_d * likelihood) / (
                            self.clutter_density + r * self.p_d * likelihood
                        )
                        updated_targets.append(
                            BernoulliTarget(x_upd, P_upd, existence_prob=r_det)
                        )

        # 2. Birth hypotheses from measurements (new target initiation)
        for j in range(M):
            meas = z[j].reshape(-1, 1)
            # Inverse measurement initialization
            x_birth = np.zeros((self.F.shape[0], 1))
            x_birth[: meas.shape[0]] = meas
            P_birth = np.eye(self.F.shape[0]) * 5.0
            updated_targets.append(
                BernoulliTarget(x_birth, P_birth, existence_prob=self.birth_rate)
            )

        # 3. Prune low-probability targets
        self.targets = [
            t for t in updated_targets if t.existence_prob >= self.prune_threshold
        ]

        # Return confirmed targets (r >= 0.5)
        return [t for t in self.targets if t.existence_prob >= 0.5]
