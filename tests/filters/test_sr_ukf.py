import numpy as np

from kalbee import SquareRootUKF


class TestSquareRootUKF:
    """Tests for Square-Root UKF."""

    def setup_method(self):
        self.state = np.zeros((2, 1))
        self.cov = np.eye(2) * 10.0
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.5]])

        def f(x, dt):
            return np.array([[x[0, 0] + x[1, 0] * dt], [x[1, 0]]])

        def h(x):
            return np.array([[x[0, 0]]])

        self.f = f
        self.h = h

    def test_predict_update(self):
        srukf = SquareRootUKF(self.state, self.cov, self.Q, self.R, self.f, self.h)
        srukf.predict(dt=1.0)
        srukf.update(np.array([[1.0]]))

        assert srukf.x.shape == (2, 1)
        assert srukf.P.shape == (2, 2)
