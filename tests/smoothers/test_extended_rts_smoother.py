import numpy as np

from kalbee import ExtendedRTSSmoother, KalmanFilter


class TestExtendedRTSSmoother:
    """Tests for Extended RTS Smoother."""

    def setup_method(self):
        self.state = np.zeros((2, 1))
        self.cov = np.eye(2) * 10.0
        self.F = np.array([[1, 1], [0, 1]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1, 0]])
        self.R = np.array([[0.5]])

    def test_smooth_empty(self):
        smoothed = ExtendedRTSSmoother.smooth([], [], [], [])
        assert smoothed == ([], [])

    def test_smooth_single_step(self):
        filtered_states = [self.state.copy()]
        filtered_covs = [self.cov.copy()]
        predicted_states = [self.state.copy()]
        predicted_covs = [self.cov.copy()]

        smoothed_s, smoothed_p = ExtendedRTSSmoother.smooth(
            filtered_states, filtered_covs, predicted_states, predicted_covs
        )
        assert len(smoothed_s) == 1

    def test_smooth_multi_step(self):
        kf = KalmanFilter(self.state, self.cov, self.F, self.Q, self.H, self.R)
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]

        filtered_states = []
        filtered_covs = []
        predicted_states = []
        predicted_covs = []

        for z in measurements:
            predicted_states.append(kf.x.copy())
            predicted_covs.append(kf.P.copy())
            kf.predict()
            kf.update(np.array([[z]]))
            filtered_states.append(kf.x.copy())
            filtered_covs.append(kf.P.copy())

        smoothed_s, smoothed_p = ExtendedRTSSmoother.smooth(
            filtered_states, filtered_covs, predicted_states, predicted_covs
        )
        assert len(smoothed_s) == len(measurements)

    def test_compute_jacobian(self):
        def f(x, dt):
            return np.array([[x[0, 0] + x[1, 0] * dt], [x[1, 0]]])

        state = np.array([[1.0], [2.0]])
        F = ExtendedRTSSmoother._compute_jacobian(state, f, dt=1.0, n=2)
        expected = np.array([[1, 1], [0, 1]])
        np.testing.assert_array_almost_equal(F, expected, decimal=4)
