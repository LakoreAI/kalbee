import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter


def test_kf_initialization():
    state = np.array([[0], [0]])
    covariance = np.eye(2)
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1, 0]])
    R = np.eye(1) * 0.5

    kf = KalmanFilter(state, covariance, F, Q, H, R)

    assert np.array_equal(kf.state, state)
    assert np.array_equal(kf.covariance, covariance)
    assert np.array_equal(kf.x, state)
    assert np.array_equal(kf.P, covariance)


def test_kf_predict():
    # Constant velocity model
    dt = 1.0
    state = np.array([[0.0], [1.0]])
    covariance = np.eye(2)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.zeros((2, 2))  # No noise for predictable test
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    kf = KalmanFilter(state, covariance, F, Q, H, R)
    kf.predict()

    # Expected state: [0 + 1*1, 1] = [1, 1]
    expected_state = np.array([[1.0], [1.0]])
    assert np.allclose(kf.state, expected_state)
    # P = FPF' + Q = F I F' = F F'
    expected_P = F @ F.T
    assert np.allclose(kf.covariance, expected_P)


def test_kf_update():
    state = np.array([[0.0], [0.0]])
    covariance = np.eye(2) * 10  # High initial uncertainty
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])  # Low measurement noise

    kf = KalmanFilter(state, covariance, F, Q, H, R)

    # Measurement at 5.0
    kf.update(np.array([[5.0]]))

    # State should move towards 5.0
    assert kf.state[0, 0] > 0
    assert kf.state[0, 0] < 5.0  # But not all the way immediately if R > 0
    # Covariance should decrease
    assert np.trace(kf.covariance) < np.trace(covariance)


def test_kf_measure():
    state = np.array([[2.0], [3.0]])
    covariance = np.eye(2)
    F = np.eye(2)
    Q = np.eye(2)
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    kf = KalmanFilter(state, covariance, F, Q, H, R)
    measurement = kf.measure()

    assert measurement.shape == (1, 1)
    assert measurement[0, 0] == 2.0
