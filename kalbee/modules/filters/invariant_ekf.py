from typing import Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class SO3:
    """
    Lie Group SO(3): Special Orthogonal Group for 3D Rotations.
    """

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix operator (vee -> hat)."""
        v = v.flatten()
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

    @staticmethod
    def vee(m: np.ndarray) -> np.ndarray:
        """Vee operator (skew-symmetric matrix -> vector)."""
        return np.array([[m[2, 1]], [m[0, 2]], [m[1, 0]]])

    @staticmethod
    def exp(phi: np.ndarray) -> np.ndarray:
        """Rodrigues' formula for matrix exponential on SO(3)."""
        phi = phi.flatten()
        angle = np.linalg.norm(phi)
        if angle < 1e-8:
            return np.eye(3) + SO3.hat(phi)

        axis = phi / angle
        K = SO3.hat(axis)
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    @staticmethod
    def log(R: np.ndarray) -> np.ndarray:
        """Matrix logarithm on SO(3)."""
        tr = np.trace(R)
        cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < 1e-8:
            return SO3.vee(R - R.T) / 2.0

        return SO3.vee((angle / (2.0 * np.sin(angle))) * (R - R.T))


class SE3:
    """
    Lie Group SE(3): Special Euclidean Group for 3D Poses.
    """

    @staticmethod
    def exp(xi: np.ndarray) -> np.ndarray:
        """Matrix exponential mapping se(3) -> SE(3)."""
        xi = xi.flatten()
        rho = xi[:3]
        phi = xi[3:]
        angle = np.linalg.norm(phi)

        R = SO3.exp(phi)
        if angle < 1e-8:
            V = np.eye(3)
        else:
            K = SO3.hat(phi / angle)
            V = np.eye(3) + ((1.0 - np.cos(angle)) / angle) * K + ((angle - np.sin(angle)) / angle) * (K @ K)

        t = V @ rho
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def adjoint(T: np.ndarray) -> np.ndarray:
        """Adjoint representation Ad_T of SE(3)."""
        R = T[:3, :3]
        t = T[:3, 3]
        t_hat = SO3.hat(t)

        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[:3, 3:] = t_hat @ R
        Ad[3:, 3:] = R
        return Ad


class InvariantEKF(BaseFilter):
    """
    Right-Invariant Extended Kalman Filter (InEKF) on SE(3).

    Operates on Matrix Lie Group SE(3) with log-linear error dynamics and
    state-independent covariance propagation.
    """

    def __init__(
        self,
        initial_pose: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
    ):
        """
        Args:
            initial_pose: 4x4 SE(3) transformation matrix.
            initial_covariance: 6x6 Lie algebra covariance matrix.
            process_noise: 6x6 Q matrix.
            measurement_noise: 3x3 or 6x6 R matrix.
        """
        pose = initial_pose if initial_pose is not None else np.eye(4)
        cov = initial_covariance if initial_covariance is not None else np.eye(6) * 0.1

        super().__init__(state=pose, covariance=cov)

        self.Q = process_noise if process_noise is not None else np.eye(6) * 0.01
        self.R = measurement_noise if measurement_noise is not None else np.eye(3) * 0.1

    @property
    def pose(self) -> np.ndarray:
        """Current 4x4 SE(3) pose matrix."""
        return self.state

    @property
    def rotation(self) -> np.ndarray:
        """Current 3x3 rotation matrix SO(3)."""
        return self.state[:3, :3]

    @property
    def position(self) -> np.ndarray:
        """Current 3D position vector (3, 1)."""
        return self.state[:3, 3].reshape(-1, 1)

    def predict(self, dt: float = 1.0, twist: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Right-Invariant predict step.

        T_pred = T * exp((twist * dt)^hat)
        P_pred = Ad * P * Ad^T + Q
        """
        if twist is None:
            twist = np.zeros(6)

        u = np.asarray(twist, dtype=float).flatten() * dt
        T_delta = SE3.exp(u)

        # Propagate state on Lie Group
        self.state = self.state @ T_delta

        # Propagate covariance in Lie algebra using Adjoint
        Ad = SE3.adjoint(T_delta)
        self.covariance = Ad @ self.covariance @ Ad.T + self.Q
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Right-Invariant update step with 3D position observation z (3x1).

        y = z - T_pos
        """
        z = np.asarray(measurement, dtype=float).reshape(3, 1)
        t_curr = self.position

        # Innovation in position space
        y = z - t_curr

        # Measurement Jacobian H (3 x 6)
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)

        P = self.covariance
        S = H @ P @ H.T + self.R
        K = P @ H.T @ safe_inv(S)

        # Compute correction vector in Lie algebra (6x1)
        dx = K @ y

        # Update pose on Lie Group using exponential map
        T_corr = SE3.exp(dx)
        self.state = T_corr @ self.state

        # Update covariance
        I_KH = np.eye(6) - K @ H
        self.covariance = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        return self.state
