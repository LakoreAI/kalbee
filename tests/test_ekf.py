import numpy as np
import pytest
from kalbee.modules.filters.ekf_filter import ExtendedKalmanFilter


def test_ekf_initialization():
    state = np.array([[0.0]])
    cov = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.1]])

    ekf = ExtendedKalmanFilter(state, cov, Q, R)
    assert np.array_equal(ekf.state, state)
    assert np.array_equal(ekf.covariance, cov)


def test_ekf_predict_linear_equivalent():
    # Setup EKF to behave like a linear KF for testing
    # x = x + v*dt (where v=1 constant)
    state = np.array([[0.0]])
    cov = np.eye(1)
    Q = np.zeros((1, 1))

    def f(x, dt):
        return x + 1.0 * dt

    def F(x, dt):
        return np.array([[1.0]])

    ekf = ExtendedKalmanFilter(state, cov, Q, np.eye(1))
    ekf.predict(dt=1.0, f=f, F=F)

    assert ekf.state[0, 0] == 1.0
    # P = FPF' + Q = 1*1*1 + 0 = 1
    assert ekf.covariance[0, 0] == 1.0


def test_ekf_update_nonlinear():
    # Measurement is z = x^2
    state = np.array([[2.0]])
    cov = np.eye(1)
    Q = np.eye(1)
    R = np.eye(1) * 0.1

    def h(x):
        return x**2

    def H_jac(x):
        # Derivative of x^2 is 2x
        return np.array([[2 * x[0, 0]]])

    ekf = ExtendedKalmanFilter(state, cov, Q, R)

    # Measurement z = 4.2 (true state approx 2.05)
    ekf.update(np.array([[4.2]]), h=h, H=H_jac)

    # Check if state updated correctly
    # Predicted Measurement = 2^2 = 4
    # Residual = 4.2 - 4 = 0.2
    # H = 4
    # S = HPH' + R = 4*1*4 + 0.1 = 16.1
    # K = PH'S^-1 = 1*4 / 16.1 = 0.248
    # New state = 2 + 0.248 * 0.2 = 2.0496

    expected_state = 2.049689
    assert np.isclose(ekf.state[0, 0], expected_state, atol=1e-3)


def test_ekf_methods_missing_args():
    state = np.array([[0.0]])
    ekf = ExtendedKalmanFilter(state, np.eye(1), np.eye(1), np.eye(1))

    with pytest.raises(ValueError):
        ekf.predict()

    with pytest.raises(ValueError):
        ekf.update(np.array([[1.0]]))
