import numpy as np

from kalbee.modules.filters.square_root_kf import SquareRootKalmanFilter


def test_srkf_initialization():
    state = np.array([[0.0], [0.0]])
    covariance = np.eye(2)
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.eye(1) * 0.5

    srkf = SquareRootKalmanFilter(state, covariance, F, Q, H, R)

    assert np.array_equal(srkf.state, state)
    assert np.allclose(srkf.covariance, covariance)
    assert np.array_equal(srkf.x, state)
    assert np.allclose(srkf.P, covariance)
    # Check that S_P is lower triangular Cholesky factor
    assert np.allclose(srkf.S_P @ srkf.S_P.T, covariance)
    assert np.allclose(np.tril(srkf.S_P), srkf.S_P)


def test_srkf_predict():
    dt = 1.0
    state = np.array([[0.0], [1.0]])
    covariance = np.eye(2)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.eye(2) * 0.01  # process noise
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    srkf = SquareRootKalmanFilter(state, covariance, F, Q, H, R)
    srkf.predict()

    # Expected state: F @ state
    expected_state = F @ state
    assert np.allclose(srkf.state, expected_state)

    # Expected covariance: F P F.T + Q
    expected_P = F @ covariance @ F.T + Q
    assert np.allclose(srkf.covariance, expected_P)
    # Check Cholesky factor consistency
    assert np.allclose(srkf.S_P @ srkf.S_P.T, expected_P)


def test_srkf_update():
    state = np.array([[0.0], [0.0]])
    covariance = np.eye(2) * 10.0  # High initial uncertainty
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])  # Low measurement noise

    srkf = SquareRootKalmanFilter(state, covariance, F, Q, H, R)

    # Measurement at 5.0
    srkf.update(np.array([[5.0]]))

    # State should move towards 5.0
    assert srkf.state[0, 0] > 0
    assert srkf.state[0, 0] < 5.0  # But not all the way immediately if R > 0
    # Covariance should decrease
    assert np.trace(srkf.covariance) < np.trace(covariance)
    # Check Cholesky factor consistency
    assert np.allclose(srkf.S_P @ srkf.S_P.T, srkf.covariance)
