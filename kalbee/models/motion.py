"""
Ready-made linear motion models (transition matrix ``F`` and process noise ``Q``).

Every model returns a plain ``(F, Q)`` pair sized for ``n_dims`` spatial axes,
using the per-axis block state convention documented in
:mod:`kalbee.models.measurement`. These plug directly into ``KalmanFilter`` and
friends, and the constant-velocity / constant-turn pair is the classic
maneuvering-target combination for ``InteractingMultipleModel``.
"""

from typing import Tuple
import numpy as np
from scipy.linalg import block_diag

from kalbee.models.measurement import discrete_white_noise, _block_per_axis


def constant_velocity(
    dt: float = 1.0,
    process_var: float = 1.0,
    n_dims: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constant-velocity (CV) model. Per-axis state is ``[position, velocity]``.

    Args:
        dt: Time step.
        process_var: Process-noise variance.
        n_dims: Number of spatial axes.

    Returns:
        Tuple ``(F, Q)`` each of shape (2 * n_dims, 2 * n_dims).
    """
    f_axis = np.array([[1.0, dt], [0.0, 1.0]])
    F = _block_per_axis(f_axis, n_dims)
    q_axis = discrete_white_noise(order=1, dt=dt, var=process_var)
    Q = _block_per_axis(q_axis, n_dims)
    return F, Q


def constant_acceleration(
    dt: float = 1.0,
    process_var: float = 1.0,
    n_dims: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constant-acceleration (CA) model. Per-axis state is
    ``[position, velocity, acceleration]``.

    Args:
        dt: Time step.
        process_var: Process-noise variance.
        n_dims: Number of spatial axes.

    Returns:
        Tuple ``(F, Q)`` each of shape (3 * n_dims, 3 * n_dims).
    """
    f_axis = np.array(
        [
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    F = _block_per_axis(f_axis, n_dims)
    q_axis = discrete_white_noise(order=2, dt=dt, var=process_var)
    Q = _block_per_axis(q_axis, n_dims)
    return F, Q


def constant_turn(
    dt: float = 1.0,
    turn_rate: float = 0.1,
    process_var: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinated-turn (CT) model with a known turn rate ``omega``.

    This is a planar (2-D) model with state ordered ``[x, vx, y, vy]``. With a
    known turn rate the model is linear, so the returned ``F`` works directly
    with :class:`~kalbee.KalmanFilter`; for an unknown/estimated turn rate use
    an EKF/UKF with a nonlinear transition instead.

    Args:
        dt: Time step.
        turn_rate: Turn rate ``omega`` in radians per unit time.
        process_var: Process-noise variance applied to each axis' CV block.

    Returns:
        Tuple ``(F, Q)`` each of shape (4, 4).
    """
    w = turn_rate
    if abs(w) < 1e-9:
        # Degenerate to constant velocity as omega -> 0 (avoids 0/0).
        return constant_velocity(dt=dt, process_var=process_var, n_dims=2)

    sin_wt = np.sin(w * dt)
    cos_wt = np.cos(w * dt)

    F = np.array(
        [
            [1.0, sin_wt / w, 0.0, -(1.0 - cos_wt) / w],
            [0.0, cos_wt, 0.0, -sin_wt],
            [0.0, (1.0 - cos_wt) / w, 1.0, sin_wt / w],
            [0.0, sin_wt, 0.0, cos_wt],
        ]
    )

    # Process noise: independent CV white-noise block per axis, reordered to the
    # [x, vx, y, vy] layout.
    q_axis = discrete_white_noise(order=1, dt=dt, var=process_var)
    Q = block_diag(q_axis, q_axis)  # ordered [x, vx, y, vy]
    return F, Q


def imu_velocity_control(dt: float = 1.0, n_dims: int = 1) -> np.ndarray:
    """
    Control matrix ``B`` for feeding a world-frame accelerometer reading
    straight into a constant-velocity ``[position, velocity]`` state.

    This is the "loosely-coupled" GPS+IMU building block: predict every IMU
    tick with the ``(F, Q)`` from :func:`constant_velocity` plus this ``B``
    and control input ``u`` = gravity-compensated, world-frame acceleration
    (e.g. rotate a raw accelerometer reading into the world frame using an
    attitude estimate from :mod:`kalbee.models.attitude`, then subtract
    gravity); update on each GPS fix with
    :func:`kalbee.models.measurement.position_measurement_model`.

    Args:
        dt: Time step between IMU samples.
        n_dims: Number of spatial axes.

    Returns:
        ``B`` of shape (2 * n_dims, n_dims).
    """
    b_axis = np.array([[0.5 * dt**2], [dt]])
    return _block_per_axis(b_axis, n_dims)
