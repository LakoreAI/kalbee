import numpy as np

from kalbee import CholeskyKalmanFilter


class TestCholeskyKalmanFilter:
    """Tests for Cholesky KF."""

    def setup_method(self):
        self.state = np.zeros((2, 1))
        self.cov = np.eye(2) * 10.0
        self.F = np.array([[1, 1], [0, 1]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1, 0]])
        self.R = np.array([[0.5]])

    def test_predict_update(self):
        ckf = CholeskyKalmanFilter(self.state, self.cov, self.F, self.Q, self.H, self.R)
        ckf.predict(dt=1.0)
        ckf.update(np.array([[1.0]]))

        assert ckf.x.shape == (2, 1)
        assert ckf.P.shape == (2, 2)

    def test_covariance_positive_definite(self):
        ckf = CholeskyKalmanFilter(self.state, self.cov, self.F, self.Q, self.H, self.R)
        for _ in range(10):
            ckf.predict(dt=1.0)
            ckf.update(np.array([[1.0]]))

        eigenvalues = np.linalg.eigvalsh(ckf.P)
        assert np.all(eigenvalues > 0)
