import numpy as np
from kalbee.modules.smoothers.rts_smoother import RTSSmoother
from kalbee.modules.filters.kf_filter import KalmanFilter


def test_rts_smoother_basic():
    """RTS smoother should reduce estimation error on a constant velocity model."""
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])

    state = np.array([[0.0], [1.0]])
    cov = np.eye(2) * 10.0

    kf = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)

    # Ground truth: constant velocity x=t, v=1
    T = 20
    np.random.seed(42)

    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []

    for t in range(T):
        # Predict
        kf.predict(dt=dt)
        predicted_states.append(kf.state.copy())
        predicted_covariances.append(kf.covariance.copy())

        # Noisy measurement
        true_pos = float(t + 1)
        z = np.array([[true_pos + np.random.randn() * 1.0]])

        # Update
        kf.update(z)
        filtered_states.append(kf.state.copy())
        filtered_covariances.append(kf.covariance.copy())

    # Run smoother
    smoothed_states, smoothed_covariances = RTSSmoother.smooth(
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        F,
    )

    assert len(smoothed_states) == T
    assert len(smoothed_covariances) == T

    # Smoothed covariances should be <= filtered covariances (element-wise trace)
    for k in range(T - 1):
        assert (
            np.trace(smoothed_covariances[k])
            <= np.trace(filtered_covariances[k]) + 1e-6
        )


def test_rts_smoother_empty():
    """Smoother should handle empty input gracefully."""
    smoothed_states, smoothed_covariances = RTSSmoother.smooth(
        [], [], [], [], np.eye(2)
    )
    assert smoothed_states == []
    assert smoothed_covariances == []


def test_rts_smoother_single():
    """Smoother with single time step should return the filtered state unchanged."""
    state = np.array([[1.0], [2.0]])
    cov = np.eye(2)

    smoothed_states, smoothed_covariances = RTSSmoother.smooth(
        [state], [cov], [state], [cov], np.eye(2)
    )

    assert len(smoothed_states) == 1
    assert np.allclose(smoothed_states[0], state)
    assert np.allclose(smoothed_covariances[0], cov)
