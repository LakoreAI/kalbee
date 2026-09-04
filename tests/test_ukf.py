import numpy as np

from kalbee.modules.filters.ukf_filter import UnscentedKalmanFilter
from kalbee.modules.filters.auto_filter import AutoFilter


def test_ukf_initialization():
    state = np.zeros((2, 1))
    cov = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)

    # Dummy functions
    def f(x, dt):
        return x

    def h(x):
        return x

    ukf = UnscentedKalmanFilter(state, cov, Q, R, f, h)

    assert np.array_equal(ukf.state, state)
    assert np.array_equal(ukf.covariance, cov)
    assert ukf.n == 2


def test_ukf_predict_linear_equivalent():
    # A UKF with linear f(x) should behave roughly like a KF (within precision limits)
    state = np.array([[0.0]])
    cov = np.eye(1)
    Q = np.zeros((1, 1))
    R = np.eye(1)

    # f(x) = x + dt
    def transition(x, dt):
        return x + 1.0 * dt

    def measurement(x):
        return x

    ukf = UnscentedKalmanFilter(state, cov, Q, R, transition, measurement)
    ukf.predict(dt=1.0)

    # State should be 0 + 1 = 1
    assert np.isclose(ukf.state[0, 0], 1.0)


def test_ukf_update_nonlinear():
    # Classic example: Radar measurement (Polar to Cartesian) usually uses EKF/UKF.
    # Here, let's test a simple non-linear measurement: z = x^2

    state = np.array([[2.0]])
    cov = np.eye(1) * 0.1
    Q = np.eye(1) * 0.01
    R = np.eye(1) * 0.01

    def transition(x, dt):
        return x  # Static state

    def measurement(x):
        return x**2

    ukf = UnscentedKalmanFilter(state, cov, Q, R, transition, measurement)

    # Measurement z = 4.0 (consistent with x=2)
    ukf.predict()
    ukf.update(np.array([[4.0]]))

    # State should remain close to 2.0, covariance should decrease
    assert np.isclose(ukf.state[0, 0], 2.0, atol=0.1)
    assert ukf.covariance[0, 0] < 0.1


def test_auto_filter_ukf():
    state = np.array([[0.0]])

    def f(x, dt):
        return x

    def h(x):
        return x

    ukf = AutoFilter.from_filter(
        state,
        np.eye(1),
        np.eye(1),
        np.eye(1),
        transition_function=f,
        measurement_function=h,
        mode="ukf",
    )

    assert isinstance(ukf, UnscentedKalmanFilter)
