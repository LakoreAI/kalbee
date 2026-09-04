import numpy as np
from kalbee.modules.filters.information_filter import InformationFilter
from kalbee.modules.filters.kf_filter import KalmanFilter


def test_info_filter_initialization():
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2)
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.eye(1) * 0.5

    inf = InformationFilter(state, cov, F, Q, H, R)

    assert np.array_equal(inf.state, state)
    # Info matrix should be inverse of covariance
    assert np.allclose(inf.Y, np.linalg.inv(cov))
    # Info state should be Y @ x
    assert np.allclose(inf.y_hat, np.linalg.inv(cov) @ state)


def test_info_filter_predict():
    state = np.array([[0.0], [1.0]])
    cov = np.eye(2)
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.zeros((2, 2))
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    inf = InformationFilter(state, cov, F, Q, H, R)
    inf.predict(dt=dt)

    # Expected state: [0 + 1*1, 1] = [1, 1]
    expected_state = np.array([[1.0], [1.0]])
    assert np.allclose(inf.state, expected_state, atol=1e-6)


def test_info_filter_equivalence_with_kf():
    """Information Filter should give same results as standard KF."""
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2) * 10.0
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])

    kf = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)
    inf = InformationFilter(state.copy(), cov.copy(), F, Q, H, R)

    # Run several predict/update cycles
    measurements = [np.array([[5.0]]), np.array([[5.1]]), np.array([[4.9]])]

    for z in measurements:
        kf.predict()
        inf.predict()

        kf.update(z)
        inf.update(z)

    # States should be very close
    assert np.allclose(kf.state, inf.state, atol=1e-6)
    # Covariances should be very close
    assert np.allclose(kf.covariance, inf.covariance, atol=1e-6)


def test_info_filter_additive_update():
    """Test that the information update is additive (key property)."""
    state = np.array([[0.0]])
    cov = np.eye(1) * 10.0
    F = np.eye(1)
    Q = np.eye(1) * 0.1
    H = np.array([[1.0]])
    R = np.array([[0.5]])

    inf = InformationFilter(state, cov, F, Q, H, R)
    inf.predict()

    Y_before = inf.Y.copy()

    # Update adds information
    inf.update(np.array([[3.0]]))

    # Information matrix should increase (more information)
    assert inf.Y[0, 0] > Y_before[0, 0]


def test_info_filter_fuse():
    """Test multi-sensor fusion via the fuse() method."""
    state = np.array([[0.0]])
    cov = np.eye(1) * 10.0
    F = np.eye(1)
    Q = np.eye(1) * 0.1
    H = np.array([[1.0]])
    R = np.array([[0.5]])

    inf = InformationFilter(state, cov, F, Q, H, R)
    Y_before = inf.Y.copy()

    # Simulate external information contribution
    external_info_matrix = np.array([[2.0]])
    external_info_state = np.array([[6.0]])

    inf.fuse(external_info_matrix, external_info_state)

    # Information should increase
    assert inf.Y[0, 0] > Y_before[0, 0]
    # Covariance should decrease (more information = less uncertainty)
    assert inf.covariance[0, 0] < cov[0, 0]
