import numpy as np

from kalbee import FederatedKalmanFilter, KalmanFilter
from kalbee.models import constant_velocity, position_measurement_model


class TestFederatedKalmanFilter:
    """Tests for Federated Kalman Filter."""

    def setup_method(self):
        F, Q = constant_velocity(dt=1.0, process_var=0.01, n_dims=1)
        H, R = position_measurement_model(order=1, n_dims=1, measurement_var=0.5)
        self.F, self.Q, self.H, self.R = F, Q, H, R

    def test_basic_fusion(self):
        state = np.zeros((2, 1))
        cov = np.eye(2) * 10.0

        kf1 = KalmanFilter(state.copy(), cov.copy(), self.F, self.Q, self.H, self.R)
        kf2 = KalmanFilter(state.copy(), cov.copy(), self.F, self.Q, self.H, self.R)
        kf_global = KalmanFilter(
            state.copy(), cov.copy(), self.F, self.Q, self.H, self.R
        )

        federated = FederatedKalmanFilter([kf1, kf2], kf_global)

        federated.predict(dt=1.0)
        result = federated.update([np.array([[1.0]]), np.array([[1.2]])])

        assert result.shape == (2, 1)
        assert federated.state.shape == (2, 1)

    def test_missing_measurement(self):
        state = np.zeros((2, 1))
        cov = np.eye(2) * 10.0

        kf1 = KalmanFilter(state.copy(), cov.copy(), self.F, self.Q, self.H, self.R)
        kf2 = KalmanFilter(state.copy(), cov.copy(), self.F, self.Q, self.H, self.R)
        kf_global = KalmanFilter(
            state.copy(), cov.copy(), self.F, self.Q, self.H, self.R
        )

        federated = FederatedKalmanFilter([kf1, kf2], kf_global)

        federated.predict(dt=1.0)
        result = federated.update([np.array([[1.0]]), None])

        assert result.shape == (2, 1)
