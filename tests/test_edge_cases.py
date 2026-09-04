import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter


def test_zero_dt_prediction():
    """Predict step with dt=0 should run without crashing."""
    state = np.array([[1.0], [2.0]])
    covariance = np.eye(2)
    F = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    kf = KalmanFilter(state, covariance, F, Q, H, R)
    kf.predict(dt=0.0)

    # State shouldn't change with identity F
    assert np.allclose(kf.state, state)
    # Covariance should be F P F.T + Q = P + Q
    assert np.allclose(kf.covariance, covariance + Q)


def test_extremely_large_noise():
    """Extremely large noise covariances should be handled stably."""
    state = np.array([[1.0], [2.0]])
    covariance = np.eye(2) * 1e6
    F = np.eye(2)
    Q = np.eye(2) * 1e6
    H = np.array([[1.0, 0.0]])
    R = np.array([[1e9]])

    kf = KalmanFilter(state, covariance, F, Q, H, R)
    kf.predict()
    kf.update(np.array([[100.0]]))

    # Ensure no NaN values
    assert not np.isnan(kf.state).any()
    assert not np.isnan(kf.covariance).any()


def test_singular_innovation_covariance_fallback():
    """Tests the fallback mechanism for singular S matrix in KF updates."""
    state = np.array([[1.0], [2.0]])
    # Force covariance and process noise to be zero to cause singular innovation covariance if R is zero
    covariance = np.zeros((2, 2))
    F = np.eye(2)
    Q = np.zeros((2, 2))
    H = np.array([[1.0, 0.0]])
    R = np.zeros(
        (1, 1)
    )  # Make R zero to ensure S = H P H.T + R is completely zero (singular!)

    kf = KalmanFilter(state, covariance, F, Q, H, R)
    kf.predict()

    # This update would crash with singular S if fallback isn't implemented.
    # Fallback uses regularization / pseudoinverse.
    kf.update(np.array([[5.0]]))

    assert not np.isnan(kf.state).any()
