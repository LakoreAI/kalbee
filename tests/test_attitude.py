import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kalbee import ExtendedKalmanFilter
from kalbee.modules.utils.jacobian import numerical_jacobian
from kalbee.models.attitude import (
    quaternion_normalize,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    quaternion_angular_error,
    attitude_transition,
    attitude_transition_jacobian,
    gravity_measurement,
    gravity_measurement_jacobian,
)


def _scipy_quat(q):
    """Convert internal [w, x, y, z] to scipy's [x, y, z, w] ordering."""
    w, x, y, z = np.asarray(q, dtype=float).flatten()
    return [x, y, z, w]


def test_quaternion_multiply_matches_scipy_rotation_composition():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q1 = quaternion_normalize(rng.standard_normal(4))
        q2 = quaternion_normalize(rng.standard_normal(4))

        q_prod = quaternion_normalize(quaternion_multiply(q1, q2))

        r1 = Rotation.from_quat(_scipy_quat(q1))
        r2 = Rotation.from_quat(_scipy_quat(q2))
        expected = (r1 * r2).as_matrix()

        actual = quaternion_to_rotation_matrix(q_prod)
        assert np.allclose(actual, expected, atol=1e-8)


def test_quaternion_to_rotation_matrix_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = quaternion_normalize(rng.standard_normal(4))
        expected = Rotation.from_quat(_scipy_quat(q)).as_matrix()
        actual = quaternion_to_rotation_matrix(q)
        assert np.allclose(actual, expected, atol=1e-8)


def test_gravity_measurement_identity_quaternion():
    q_identity = np.array([[1.0], [0.0], [0.0], [0.0]])
    g = gravity_measurement(q_identity)
    assert np.allclose(g, [[0.0], [0.0], [1.0]])


def test_gravity_measurement_jacobian_matches_numerical():
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = quaternion_normalize(rng.standard_normal(4))
        analytic = gravity_measurement_jacobian(q)
        numeric = numerical_jacobian(gravity_measurement, q)
        assert np.allclose(analytic, numeric, atol=1e-5)


def test_attitude_transition_jacobian_matches_numerical():
    rng = np.random.default_rng(3)
    dt = 0.02
    for _ in range(20):
        q = quaternion_normalize(rng.standard_normal(4))
        gyro = rng.standard_normal(3)
        analytic = attitude_transition_jacobian(q, dt, gyro)
        numeric = numerical_jacobian(lambda s: attitude_transition(s, dt, gyro), q)
        assert np.allclose(analytic, numeric, atol=1e-5)


def test_attitude_transition_matches_scipy_integration():
    q0 = np.array([[1.0], [0.0], [0.0], [0.0]])
    gyro = np.array([0.1, -0.2, 0.3])
    dt = 0.01
    steps = 100

    q = q0.copy()
    for _ in range(steps):
        q = quaternion_normalize(attitude_transition(q, dt, gyro))

    # scipy: constant-angular-velocity rotation over total time = steps * dt
    total_time = steps * dt
    expected = Rotation.from_rotvec(gyro * total_time)
    actual = Rotation.from_quat(_scipy_quat(q))

    # Angular distance between the two rotations should be near zero.
    rel = expected.inv() * actual
    angle = rel.magnitude()
    assert angle < 1e-3


def test_attitude_ekf_converges_to_true_orientation():
    """
    Simulate a body spinning at a constant known rate, feed the EKF noisy
    gyro (for predict) and noisy accelerometer (for update), and check the
    filtered orientation converges close to the true one.
    """
    rng = np.random.default_rng(42)
    dt = 0.01
    steps = 400
    true_gyro = np.array([0.05, 0.02, -0.03])
    gyro_noise_std = 0.01
    accel_noise_std = 0.02

    q_true = np.array([[1.0], [0.0], [0.0], [0.0]])

    q0 = np.array([[1.0], [0.0], [0.0], [0.0]])
    P0 = np.eye(4) * 0.1
    Q = np.eye(4) * 1e-4
    # R is a tuning knob, not just "sensor noise variance squared": this is a
    # full/naive quaternion-state EKF (not an error-state MEKF), so its
    # linearization only holds for small corrections. Inflating R somewhat
    # beyond the raw accelerometer noise keeps each update small enough to
    # stay in the valid linear regime — see the module docstring.
    R = np.eye(3) * 0.05

    ekf = ExtendedKalmanFilter(
        state=q0,
        covariance=P0,
        transition_covariance=Q,
        measurement_covariance=R,
    )

    for _ in range(steps):
        q_true = quaternion_normalize(attitude_transition(q_true, dt, true_gyro))

        measured_gyro = true_gyro + rng.standard_normal(3) * gyro_noise_std
        ekf.predict(
            dt=dt,
            f=lambda x, dt: attitude_transition(x, dt, measured_gyro),
            F=lambda x, dt: attitude_transition_jacobian(x, dt, measured_gyro),
        )
        ekf.state = quaternion_normalize(ekf.state)

        accel = (
            gravity_measurement(q_true) + rng.standard_normal((3, 1)) * accel_noise_std
        )
        ekf.update(accel, h=gravity_measurement, H=gravity_measurement_jacobian)
        ekf.state = quaternion_normalize(ekf.state)

    error_deg = np.degrees(quaternion_angular_error(ekf.state, q_true))
    assert error_deg < 5.0


def test_quaternion_conjugate_is_inverse_for_unit_quaternion():
    rng = np.random.default_rng(4)
    q = quaternion_normalize(rng.standard_normal(4))
    identity = quaternion_multiply(q, quaternion_conjugate(q))
    assert np.allclose(identity, [[1.0], [0.0], [0.0], [0.0]], atol=1e-8)


def test_quaternion_angular_error_zero_for_identical_quaternions():
    q = quaternion_normalize(np.array([1.0, 0.2, -0.1, 0.05]))
    assert quaternion_angular_error(q, q) == pytest.approx(0.0, abs=1e-8)
