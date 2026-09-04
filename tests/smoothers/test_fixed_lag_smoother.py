import numpy as np

from kalbee import FixedLagSmoother, KalmanFilter
from kalbee.modules.smoothers.rts_smoother import RTSSmoother


class TestFixedLagSmoother:
    """Tests for Fixed-Lag Smoother."""

    def setup_method(self):
        self.state = np.zeros((2, 1))
        self.cov = np.eye(2) * 10.0
        self.F = np.array([[1, 1], [0, 1]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1, 0]])
        self.R = np.array([[0.5]])

    def test_init(self):
        kf = KalmanFilter(self.state, self.cov, self.F, self.Q, self.H, self.R)
        smoother = FixedLagSmoother(kf, lag=5)
        assert smoother.lag == 5
        assert smoother.n == 2

    def test_predict_update(self):
        kf = KalmanFilter(self.state, self.cov, self.F, self.Q, self.H, self.R)
        smoother = FixedLagSmoother(kf, lag=3)

        smoother.predict(dt=1.0)
        smoothed = smoother.update(np.array([[1.0]]))
        assert smoothed.shape == (2, 1)

    def test_smoothed_state_available(self):
        kf = KalmanFilter(self.state, self.cov, self.F, self.Q, self.H, self.R)
        smoother = FixedLagSmoother(kf, lag=3)

        for z in [1.0, 2.0, 3.0, 4.0, 5.0]:
            smoother.predict(dt=1.0)
            smoother.update(np.array([[z]]))

        assert smoother.smoothed_state is not None
        assert smoother.smoothed_covariance is not None

    def test_exact_rts_match(self):
        """FixedLagSmoother output must match full offline RTS Smoother."""
        dt = 1.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[1.0]])

        state = np.array([[0.0], [1.0]])
        cov = np.eye(2) * 10.0

        kf = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)
        smoother = FixedLagSmoother(kf, lag=3)

        measurements = [np.array([[float(t) + 0.1 * t]]) for t in range(5)]

        filtered_states = []
        filtered_covariances = []
        predicted_states = []
        predicted_covariances = []

        kf_ref = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)

        for z in measurements:
            kf_ref.predict()
            predicted_states.append(kf_ref.state.copy())
            predicted_covariances.append(kf_ref.covariance.copy())
            kf_ref.update(z)
            filtered_states.append(kf_ref.state.copy())
            filtered_covariances.append(kf_ref.covariance.copy())

            smoother.predict()
            smoother.update(z)

        rts_states, rts_covs = RTSSmoother.smooth(
            filtered_states,
            filtered_covariances,
            predicted_states,
            predicted_covariances,
            F,
        )

        np.testing.assert_allclose(smoother.smoothed_state, rts_states[1], atol=1e-12)
        np.testing.assert_allclose(
            smoother.smoothed_covariance, rts_covs[1], atol=1e-12
        )
