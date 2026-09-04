"""Tests for the CKF (Cubature Kalman Filter)."""

import numpy as np

from kalbee import CubatureKalmanFilter


class TestCubatureKalmanFilter:
    """Tests for Cubature Kalman Filter."""

    def setup_method(self):
        self.state = np.zeros((2, 1))
        self.cov = np.eye(2) * 10.0
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.5]])
        self.H = np.array([[1.0, 0.0]])

        def f(x, dt):
            return np.array([[x[0, 0] + x[1, 0] * dt], [x[1, 0]]])

        def h(x):
            return np.array([[x[0, 0]]])

        self.f = f
        self.h = h

    def test_init(self):
        ckf = CubatureKalmanFilter(self.state, self.cov, self.Q, self.R, self.f, self.h)
        assert ckf.n == 2
        np.testing.assert_array_equal(ckf.x, self.state)

    def test_predict(self):
        ckf = CubatureKalmanFilter(self.state, self.cov, self.Q, self.R, self.f, self.h)
        state = ckf.predict(dt=1.0)
        assert state.shape == (2, 1)
        assert ckf.covariance.shape == (2, 2)

    def test_update(self):
        ckf = CubatureKalmanFilter(self.state, self.cov, self.Q, self.R, self.f, self.h)
        ckf.predict(dt=1.0)
        state = ckf.update(np.array([[1.0]]))
        assert state.shape == (2, 1)
        assert ckf.last_y is not None
        assert ckf.last_S is not None

    def test_predict_update_cycle(self):
        ckf = CubatureKalmanFilter(self.state, self.cov, self.Q, self.R, self.f, self.h)
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        for z in measurements:
            ckf.predict(dt=1.0)
            ckf.update(np.array([[z]]))

        assert ckf.x[0, 0] > 0
        assert ckf.x[1, 0] > 0

    def test_covariance_symmetry(self):
        ckf = CubatureKalmanFilter(self.state, self.cov, self.Q, self.R, self.f, self.h)
        ckf.predict(dt=1.0)
        ckf.update(np.array([[1.0]]))
        np.testing.assert_array_almost_equal(ckf.P, ckf.P.T)
