import numpy as np

from kalbee import UnscentedKalmanFilter, UnscentedRTSSmoother


def test_unscented_rts_smoother():
    """UnscentedRTSSmoother should run backward pass on UKF trajectory."""

    def f(x, dt):
        return np.array([[x[0, 0] + x[1, 0] * dt], [x[1, 0]]])

    def h(x):
        return np.array([[x[0, 0]]])

    state = np.array([[0.0], [1.0]])
    cov = np.eye(2) * 10.0
    Q = np.eye(2) * 0.01
    R = np.array([[0.5]])

    ukf = UnscentedKalmanFilter(state.copy(), cov.copy(), Q, R, f, h)

    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []

    for t in range(5):
        ukf.predict(dt=1.0)
        predicted_states.append(ukf.state.copy())
        predicted_covariances.append(ukf.covariance.copy())

        ukf.update(np.array([[float(t + 1)]]))
        filtered_states.append(ukf.state.copy())
        filtered_covariances.append(ukf.covariance.copy())

    smoothed_states, smoothed_covs = UnscentedRTSSmoother.smooth(
        filtered_states,
        filtered_covariances,
        predicted_states,
        predicted_covariances,
        transition_function=f,
    )

    assert len(smoothed_states) == 5
    assert len(smoothed_covs) == 5
    assert smoothed_states[0].shape == (2, 1)
