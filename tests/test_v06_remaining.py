import numpy as np
import pytest
from kalbee import (
    FederatedKalmanFilter,
    RaoBlackwellizedParticleFilter,
    CholeskyKalmanFilter,
    SquareRootUKF,
    AsyncKalmanFilter,
    FilterState,
    FilterConfig,
    OnlineEM,
    track_to_track_fusion,
    KalmanFilter,
)
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
        kf_global = KalmanFilter(state.copy(), cov.copy(), self.F, self.Q, self.H, self.R)

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
        kf_global = KalmanFilter(state.copy(), cov.copy(), self.F, self.Q, self.H, self.R)

        federated = FederatedKalmanFilter([kf1, kf2], kf_global)

        federated.predict(dt=1.0)
        result = federated.update([np.array([[1.0]]), None])

        assert result.shape == (2, 1)


class TestRaoBlackwellizedParticleFilter:
    """Tests for RBPF."""

    def test_basic_rbpf(self):
        def f_linear(x_l, x_nl, dt):
            return np.array([[x_l[0, 0] + x_l[1, 0] * dt], [x_l[1, 0]]])

        def f_nonlinear(x_nl, dt):
            return np.array([[x_nl[0, 0] + np.sin(x_nl[1, 0]) * dt], [x_nl[1, 0]]])

        def h(x):
            return np.array([[x[0, 0]]])

        rbpf = RaoBlackwellizedParticleFilter(
            n_particles=50,
            linear_dim=2,
            nonlinear_dim=2,
            transition_function_linear=f_linear,
            transition_function_nonlinear=f_nonlinear,
            measurement_function=h,
            process_noise_linear=np.eye(2) * 0.01,
            process_noise_nonlinear=np.eye(2) * 0.01,
            measurement_noise=np.array([[0.5]]),
        )

        rbpf.predict(dt=1.0)
        rbpf.update(np.array([[1.0]]))

        assert rbpf.state.shape == (4, 1)
        assert rbpf.weights.sum() == pytest.approx(1.0)


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
        ckf = CholeskyKalmanFilter(
            self.state, self.cov, self.F, self.Q, self.H, self.R
        )
        ckf.predict(dt=1.0)
        ckf.update(np.array([[1.0]]))

        assert ckf.x.shape == (2, 1)
        assert ckf.P.shape == (2, 2)

    def test_covariance_positive_definite(self):
        ckf = CholeskyKalmanFilter(
            self.state, self.cov, self.F, self.Q, self.H, self.R
        )
        for _ in range(10):
            ckf.predict(dt=1.0)
            ckf.update(np.array([[1.0]]))

        eigenvalues = np.linalg.eigvalsh(ckf.P)
        assert np.all(eigenvalues > 0)


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
        srukf = SquareRootUKF(
            self.state, self.cov, self.Q, self.R, self.f, self.h
        )
        srukf.predict(dt=1.0)
        srukf.update(np.array([[1.0]]))

        assert srukf.x.shape == (2, 1)
        assert srukf.P.shape == (2, 2)


class TestAsyncKalmanFilter:
    """Tests for Async wrapper."""

    def test_basic_async(self):
        import asyncio

        state = np.zeros((2, 1))
        cov = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        kf = KalmanFilter(state, cov, F, Q, H, R)
        async_kf = AsyncKalmanFilter(kf)

        async def run():
            await async_kf.predict(dt=1.0)
            return await async_kf.update(np.array([[1.0]]))

        result = asyncio.run(run())
        assert result.shape == (2, 1)


class TestFilterState:
    """Tests for FilterState dataclass."""

    def test_basic_creation(self):
        state = FilterState(
            state_mean=np.array([[1.0], [2.0]]),
            state_covariance=np.eye(2),
        )

        assert state.state_dim == 2
        assert state.covariance_trace == 2.0
        assert state.state_std.shape == (2,)

    def test_serialization(self):
        state = FilterState(
            state_mean=np.array([[1.0], [2.0]]),
            state_covariance=np.eye(2) * 0.5,
            timestamp=100,
        )

        json_str = state.to_json()
        restored = FilterState.from_json(json_str)

        np.testing.assert_array_equal(state.state_mean, restored.state_mean)
        assert state.timestamp == restored.timestamp


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_basic_creation(self):
        config = FilterConfig(
            filter_type="kf",
            state_dim=2,
            measurement_dim=1,
            transition_matrix=np.array([[1, 1], [0, 1]]),
        )

        assert config.filter_type == "kf"
        assert config.state_dim == 2

    def test_serialization(self):
        config = FilterConfig(
            filter_type="kf",
            state_dim=2,
            measurement_dim=1,
            transition_matrix=np.array([[1, 1], [0, 1]]),
            transition_covariance=np.eye(2) * 0.01,
        )

        json_str = config.to_json()
        restored = FilterConfig.from_json(json_str)

        assert config.filter_type == restored.filter_type
        np.testing.assert_array_equal(
            config.transition_matrix, restored.transition_matrix
        )


class TestOnlineEM:
    """Tests for Online EM."""

    def test_basic_update(self):
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])

        em = OnlineEM(F, H, forgetting_factor=0.99)

        state = np.array([[1.0], [0.5]])
        cov = np.eye(2) * 0.1
        pred = np.array([[1.5], [0.5]])
        pred_cov = np.eye(2) * 0.2
        z = np.array([[1.2]])

        Q, R = em.update(state, cov, pred, pred_cov, z)

        assert Q.shape == (2, 2)
        assert R.shape == (1, 1)
        assert em.sample_count == 1


class TestTrackToTrackFusion:
    """Tests for track-to-track fusion."""

    def test_ci_fusion(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2)
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2

        fused_mean, fused_cov = track_to_track_fusion(
            mean_a, cov_a, mean_b, cov_b, method="ci"
        )

        assert fused_mean.shape == (2, 1)
        assert fused_cov.shape == (2, 2)

    def test_simple_fusion(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2)
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2

        fused_mean, fused_cov = track_to_track_fusion(
            mean_a, cov_a, mean_b, cov_b, method="simple"
        )

        assert fused_mean.shape == (2, 1)
