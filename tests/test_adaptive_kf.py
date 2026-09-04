import numpy as np
from kalbee.modules.filters.adaptive_kf import AdaptiveKalmanFilter


def test_adaptive_kf_initialization():
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2)
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.eye(1) * 0.5

    akf = AdaptiveKalmanFilter(state, cov, F, Q, H, R, window_size=10)

    assert akf.window_size == 10
    assert akf.adapt_Q is True
    assert akf.adapt_R is True
    assert len(akf._innovations) == 0


def test_adaptive_kf_predict_same_as_kf():
    """Predict step should be identical to standard KF."""
    state = np.array([[0.0], [1.0]])
    cov = np.eye(2)
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.zeros((2, 2))
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    akf = AdaptiveKalmanFilter(state, cov, F, Q, H, R)
    akf.predict()

    expected_state = np.array([[1.0], [1.0]])
    assert np.allclose(akf.state, expected_state)


def test_adaptive_kf_update_stores_innovations():
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2) * 10.0
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])

    akf = AdaptiveKalmanFilter(state, cov, F, Q, H, R, window_size=5)

    for i in range(3):
        akf.predict()
        akf.update(np.array([[float(i)]]))

    assert len(akf._innovations) == 3
    assert len(akf.get_innovation_history()) == 3


def test_adaptive_kf_window_trimming():
    state = np.array([[0.0]])
    cov = np.eye(1) * 10.0
    F = np.eye(1)
    Q = np.eye(1) * 0.1
    H = np.array([[1.0]])
    R = np.array([[0.1]])

    akf = AdaptiveKalmanFilter(state, cov, F, Q, H, R, window_size=3)

    for i in range(10):
        akf.predict()
        akf.update(np.array([[float(i)]]))

    # Window should be trimmed to 3
    assert len(akf._innovations) == 3


def test_adaptive_kf_convergence():
    """AKF should still converge to a good estimate."""
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 100.0
    F = np.eye(1)
    Q = np.eye(1) * 0.01
    H = np.array([[1.0]])
    R = np.array([[0.5]])

    akf = AdaptiveKalmanFilter(state, cov, F, Q, H, R, window_size=10)

    true_state = 5.0
    for _ in range(50):
        akf.predict()
        z = np.array([[true_state + np.random.randn() * 0.5]])
        akf.update(z)

    assert np.isclose(akf.state[0, 0], true_state, atol=1.0)
