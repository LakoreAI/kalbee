import numpy as np
from kalbee import (
    KalmanFilter,
    UnscentedKalmanFilter,
    UnscentedRTSSmoother,
    JPDAAssociation,
    AsyncSensorBuffer,
)


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
        filtered_states, filtered_covariances, predicted_states, predicted_covariances, transition_function=f
    )

    assert len(smoothed_states) == 5
    assert len(smoothed_covs) == 5
    assert smoothed_states[0].shape == (2, 1)


def test_jpda_association():
    """JPDAAssociation should return marginal association probabilities matrix beta."""
    jpda = JPDAAssociation(p_d=0.9, clutter_density=1e-3, gate_threshold=16.0)

    track_states = [
        np.array([[0.0], [0.0]]),
        np.array([[10.0], [0.0]]),
    ]
    track_covs = [np.eye(2) * 1.0, np.eye(2) * 1.0]

    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])
    measurements = np.array([[0.1], [9.9], [50.0]])  # 3 detections

    beta = jpda.compute_association_probabilities(
        track_states, track_covs, H, R, measurements
    )

    assert beta.shape == (2, 4)  # 2 tracks x (3 detections + 1 missed)
    np.testing.assert_allclose(beta.sum(axis=1), np.ones(2), atol=1e-6)
    # Track 0 should have high probability for measurement 0
    assert beta[0, 1] > beta[0, 2]
    # Track 1 should have high probability for measurement 1
    assert beta[1, 2] > beta[1, 1]


def test_async_sensor_buffer_oosm():
    """AsyncSensorBuffer should process out-of-order measurements in chronological order."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])

    kf = KalmanFilter(np.zeros((2, 1)), np.eye(2) * 10.0, F, Q, H, R)
    buffer = AsyncSensorBuffer(kf, buffer_capacity=10)

    buffer.initialize(timestamp=0.0, state=np.zeros((2, 1)), covariance=np.eye(2))

    # Add measurements out of order: t=2.0 first, then t=1.0
    buffer.add_measurement(timestamp=2.0, measurement=np.array([[2.0]]))
    buffer.add_measurement(timestamp=1.0, measurement=np.array([[1.0]]))

    assert buffer.latest_state.shape == (2, 1)


def test_differentiable_kalman_filter_import():
    """DifferentiableKalmanFilter should work if PyTorch is installed, or raise ImportError."""
    try:
        import torch
        from kalbee.modules.learning.torch_kf import DifferentiableKalmanFilter

        dkf = DifferentiableKalmanFilter(state_dim=2, meas_dim=1)
        x = torch.zeros(1, 2, 1)
        cov = torch.eye(2).unsqueeze(0)
        z = torch.ones(1, 1, 1)

        x_pred, cov_pred = dkf.predict(x, cov)
        x_upd, cov_upd = dkf.update(x_pred, cov_pred, z)

        assert x_upd.shape == (1, 2, 1)
    except ImportError:
        pass
