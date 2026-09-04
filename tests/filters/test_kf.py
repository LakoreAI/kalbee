import numpy as np
import pytest

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


# ============================================================
# Control inputs
# ============================================================


class TestKFControlInput:
    def test_predict_with_control(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            control_matrix=np.array([[0.5], [0.1]]),
        )
        kf.predict(u=np.array([[2.0]]))
        # x = F @ x + B @ u = [[0,1],[0,0]] @ [0,0] + [[0.5],[0.1]] @ [2] = [1.0, 0.2]
        assert kf.state[0, 0] == pytest.approx(1.0)
        assert kf.state[1, 0] == pytest.approx(0.2)

    def test_predict_without_control(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            control_matrix=np.array([[0.5], [0.1]]),
        )
        kf.predict()
        # x = F @ x = [[1,1],[0,1]] @ [0,0] = [0, 0]
        assert kf.state[0, 0] == pytest.approx(0.0)
        assert kf.state[1, 0] == pytest.approx(0.0)

    def test_predict_with_B_in_kwargs(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        B = np.array([[0.5], [0.1]])
        kf.predict(u=np.array([[2.0]]), B=B)
        assert kf.state[0, 0] == pytest.approx(1.0)


# ============================================================
# Edge cases / numerical robustness
# ============================================================


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
