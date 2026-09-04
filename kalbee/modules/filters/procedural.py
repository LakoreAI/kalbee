from typing import Tuple, Optional
import numpy as np


def kf_predict(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    B: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman Filter predict step (procedural).

    Args:
        x: State estimate (n x 1).
        P: State covariance (n x n).
        F: Transition matrix (n x n).
        Q: Process noise covariance (n x n).
        B: Control input matrix (n x k). Optional.
        u: Control input vector (k x 1). Optional.

    Returns:
        Tuple of (predicted_state, predicted_covariance).
    """
    x_pred = F @ x
    if B is not None and u is not None:
        x_pred = x_pred + B @ u

    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred


def kf_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kalman Filter update step (procedural) using Joseph form.

    Args:
        x: Predicted state (n x 1).
        P: Predicted covariance (n x n).
        z: Measurement (m x 1).
        H: Measurement matrix (m x n).
        R: Measurement noise covariance (m x m).

    Returns:
        Tuple of (updated_state, updated_covariance, innovation, innovation_covariance).
    """
    # Innovation
    y = z - H @ x

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # Updated state
    x_upd = x + K @ y

    # Updated covariance (Joseph form)
    identity = np.eye(len(x))
    I_KH = identity - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T

    # Enforce symmetry
    P_upd = (P_upd + P_upd.T) / 2.0

    return x_upd, P_upd, y, S


def ekf_predict(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    B: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended Kalman Filter predict step (procedural).

    Same as KF predict since the transition is linear in EKF.

    Args:
        x: State estimate (n x 1).
        P: State covariance (n x n).
        F: Transition matrix (n x n) or Jacobian.
        Q: Process noise covariance (n x n).
        B: Control input matrix (n x k). Optional.
        u: Control input vector (k x 1). Optional.

    Returns:
        Tuple of (predicted_state, predicted_covariance).
    """
    return kf_predict(x, P, F, Q, B, u)


def ekf_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extended Kalman Filter update step (procedural).

    Same as KF update since the measurement model is linear in EKF.

    Args:
        x: Predicted state (n x 1).
        P: Predicted covariance (n x n).
        z: Measurement (m x 1).
        H: Measurement Jacobian (m x n).
        R: Measurement noise covariance (m x m).

    Returns:
        Tuple of (updated_state, updated_covariance, innovation, innovation_covariance).
    """
    return kf_update(x, P, z, H, R)


def compute_kalman_gain(
    P: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Compute Kalman gain matrix.

    Args:
        P: Covariance matrix (n x n).
        H: Measurement matrix (m x n).
        R: Measurement noise covariance (m x m).

    Returns:
        Kalman gain K (n x m).
    """
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return K


def compute_nis(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
) -> float:
    """
    Compute Normalized Innovation Squared.

    Args:
        innovation: Innovation vector (m x 1).
        innovation_covariance: Innovation covariance (m x m).

    Returns:
        NIS value (scalar).
    """
    return float(innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)


def compute_nees(
    state_error: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Compute Normalized Estimation Error Squared.

    Args:
        state_error: State error vector (n x 1).
        covariance: State covariance (n x n).

    Returns:
        NEES value (scalar).
    """
    return float(state_error.T @ np.linalg.inv(covariance) @ state_error)
