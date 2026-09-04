"""
Ready-made building blocks for a quaternion-based attitude (orientation) EKF.

Fuses a gyroscope (angular velocity) with an accelerometer (gravity direction)
to estimate 3D orientation — the standard first stage of most IMU/AHRS
pipelines. The state is a unit quaternion ``q = [w, x, y, z]`` (Hamilton
convention, scalar-first, body-to-world).

Plug the functions below into :class:`~kalbee.ExtendedKalmanFilter`::

    from kalbee import ExtendedKalmanFilter
    from kalbee.models.attitude import (
        attitude_transition, attitude_transition_jacobian,
        gravity_measurement, gravity_measurement_jacobian,
    )

    ekf = ExtendedKalmanFilter(
        state=q0, covariance=P0,
        transition_covariance=Q, measurement_covariance=R,
    )
    ekf.predict(
        dt=dt,
        f=lambda x, dt: attitude_transition(x, dt, gyro),
        F=lambda x, dt: attitude_transition_jacobian(x, dt, gyro),
    )
    ekf.state = quaternion_normalize(ekf.state)  # renormalize after every step
    ekf.update(accel_reading, h=gravity_measurement, H=gravity_measurement_jacobian)
    ekf.state = quaternion_normalize(ekf.state)

Because ``q_{k+1} = q_k ⊗ dq`` is linear in ``q_k`` for a fixed,
gyro-derived delta quaternion ``dq``, :func:`attitude_transition_jacobian` is
*exact* — there is no first-order linearization error in the process model,
unlike most EKF transition Jacobians.

Two caveats worth knowing before you tune this:

1. Gravity alone cannot observe rotation about the vertical (yaw) axis — any
   two orientations that differ only by a yaw rotation produce the same
   predicted accelerometer reading. Gyro noise/bias on the yaw axis therefore
   random-walks uncorrected; add a magnetometer (or GPS-course) update to fix
   yaw too.
2. This is a *naive* quaternion-state EKF (the raw 4-vector is the state),
   not an error-state/multiplicative EKF (MEKF). Its linearization only
   holds for small corrections, so treat ``R`` as a tuning knob rather than
   literally the accelerometer's noise variance: an ``R`` that's too tight
   relative to how "wrong" the initial estimate can get lets a single update
   take too large a step off the unit-quaternion manifold before
   renormalizing, which degrades convergence. Inflating ``R`` a few times
   past the raw sensor spec (or inflating ``P0``/``Q`` more gently) keeps
   updates small and the filter well behaved — see the test suite for
   worked-through numbers.
"""

import numpy as np


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion (4, 1) or (4,) to unit norm."""
    q = np.asarray(q, dtype=float).reshape(4, 1)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([[1.0], [0.0], [0.0], [0.0]])
    return q / norm


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse, for unit quaternions) of ``q``."""
    w, x, y, z = np.asarray(q, dtype=float).flatten()
    return np.array([[w], [-x], [-y], [-z]])


def quaternion_left_matrix(q: np.ndarray) -> np.ndarray:
    """4x4 matrix ``L(q)`` such that ``q ⊗ p == L(q) @ p`` (Hamilton product)."""
    w, x, y, z = np.asarray(q, dtype=float).flatten()
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def quaternion_right_matrix(q: np.ndarray) -> np.ndarray:
    """4x4 matrix ``R(q)`` such that ``p ⊗ q == R(q) @ p`` (Hamilton product)."""
    w, x, y, z = np.asarray(q, dtype=float).flatten()
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, z, -y],
            [y, -z, w, x],
            [z, y, -x, w],
        ]
    )


def quaternion_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product ``q ⊗ p``, both (4, 1)."""
    return quaternion_left_matrix(q) @ np.asarray(p, dtype=float).reshape(4, 1)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix for unit quaternion ``q = [w, x, y, z]``."""
    w, x, y, z = np.asarray(q, dtype=float).flatten()
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def quaternion_angular_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle (radians) between two orientation quaternions."""
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    dq = quaternion_multiply(quaternion_conjugate(q1), q2)
    w = float(np.clip(abs(dq[0, 0]), -1.0, 1.0))
    return 2.0 * np.arccos(w)


def _delta_quaternion(gyro: np.ndarray, dt: float) -> np.ndarray:
    """Exact integral of a constant body-frame angular velocity over ``dt``."""
    gyro = np.asarray(gyro, dtype=float).flatten()
    angle = np.linalg.norm(gyro) * dt
    if angle < 1e-12:
        return np.array([[1.0], [0.0], [0.0], [0.0]])
    axis = gyro / np.linalg.norm(gyro)
    half = angle / 2.0
    return np.array(
        [
            [np.cos(half)],
            [axis[0] * np.sin(half)],
            [axis[1] * np.sin(half)],
            [axis[2] * np.sin(half)],
        ]
    )


def attitude_transition(x: np.ndarray, dt: float, gyro: np.ndarray) -> np.ndarray:
    """
    Predict the next orientation quaternion given a (constant-over-``dt``)
    body-frame angular velocity ``gyro`` (rad/s), e.g. a raw gyroscope reading.

    Assumes ``x`` is already (approximately) unit norm. Deliberately does
    *not* renormalize internally — for the EKF process Jacobian
    (:func:`attitude_transition_jacobian`) to be exact rather than a
    first-order approximation, ``f`` must not be entangled with the
    normalization step. Renormalize the filter state explicitly instead,
    e.g. ``ekf.state = quaternion_normalize(ekf.state)`` after each predict.
    """
    dq = _delta_quaternion(gyro, dt)
    return quaternion_multiply(x, dq)


def attitude_transition_jacobian(
    x: np.ndarray, dt: float, gyro: np.ndarray
) -> np.ndarray:
    """Exact Jacobian of :func:`attitude_transition` w.r.t. the state quaternion."""
    dq = _delta_quaternion(gyro, dt)
    return quaternion_right_matrix(dq)


def gravity_measurement(x: np.ndarray) -> np.ndarray:
    """
    Predicted (unit) accelerometer reading in the body frame — the world "up"
    vector ``[0, 0, 1]`` rotated into the body frame, i.e. what an
    accelerometer reads at rest.

    Assumes ``x`` is (approximately) unit norm; see the note on
    :func:`attitude_transition` about why no internal renormalization happens
    here.
    """
    R = quaternion_to_rotation_matrix(x)
    return R.T @ np.array([[0.0], [0.0], [1.0]])


def gravity_measurement_jacobian(x: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of :func:`gravity_measurement` w.r.t. the state quaternion."""
    w, x_, y, z = np.asarray(x, dtype=float).flatten()
    return 2.0 * np.array(
        [
            [-y, z, -w, x_],
            [x_, w, z, y],
            [0.0, -2 * x_, -2 * y, 0.0],
        ]
    )
