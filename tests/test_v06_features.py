import numpy as np
import pytest
from kalbee import (
    CubatureKalmanFilter,
    ExtendedRTSSmoother,
    FixedLagSmoother,
    covariance_intersection,
    sequential_covariance_intersection,
    kf_predict,
    kf_update,
    compute_kalman_gain,
    compute_nis,
    compute_nees,
    KalmanFilter,
)


class TestCubatureKalmanFilter:
    """Tests for Cubature Kalman Filter."""

    def setup_method(self):
        self.state = np.zeros((2, 1))
        self.cov = np.eye(2) * 10.0
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.5]])
        self.H = np.array([[1.0, 0.0]])

        def f(x, dt):
            return np.array([
                [x[0, 0] + x[1, 0] * dt],
                [x[1, 0]]
            ])

        def h(x):
            return np.array([[x[0, 0]]])

        self.f = f
        self.h = h

    def test_init(self):
        ckf = CubatureKalmanFilter(
            self.state, self.cov, self.Q, self.R, self.f, self.h
        )
        assert ckf.n == 2
        np.testing.assert_array_equal(ckf.x, self.state)

    def test_predict(self):
        ckf = CubatureKalmanFilter(
            self.state, self.cov, self.Q, self.R, self.f, self.h
        )
        state = ckf.predict(dt=1.0)
        assert state.shape == (2, 1)
        assert ckf.covariance.shape == (2, 2)

    def test_update(self):
        ckf = CubatureKalmanFilter(
            self.state, self.cov, self.Q, self.R, self.f, self.h
        )
        ckf.predict(dt=1.0)
        state = ckf.update(np.array([[1.0]]))
        assert state.shape == (2, 1)
        assert ckf.last_y is not None
        assert ckf.last_S is not None

    def test_predict_update_cycle(self):
        ckf = CubatureKalmanFilter(
            self.state, self.cov, self.Q, self.R, self.f, self.h
        )
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        for z in measurements:
            ckf.predict(dt=1.0)
            ckf.update(np.array([[z]]))

        assert ckf.x[0, 0] > 0
        assert ckf.x[1, 0] > 0

    def test_covariance_symmetry(self):
        ckf = CubatureKalmanFilter(
            self.state, self.cov, self.Q, self.R, self.f, self.h
        )
        ckf.predict(dt=1.0)
        ckf.update(np.array([[1.0]]))
        np.testing.assert_array_almost_equal(ckf.P, ckf.P.T)


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
            return np.array([
                [x[0, 0] + x[1, 0] * dt],
                [x[1, 0]]
            ])

        state = np.array([[1.0], [2.0]])
        F = ExtendedRTSSmoother._compute_jacobian(state, f, dt=1.0, n=2)
        expected = np.array([[1, 1], [0, 1]])
        np.testing.assert_array_almost_equal(F, expected, decimal=4)


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


class TestCovarianceIntersection:
    """Tests for Covariance Intersection."""

    def test_basic_fusion(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2) * 1.0
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2.0

        mean_fused, cov_fused = covariance_intersection(mean_a, cov_a, mean_b, cov_b)

        assert mean_fused.shape == (2, 1)
        assert cov_fused.shape == (2, 2)
        np.testing.assert_array_almost_equal(cov_fused, cov_fused.T)

    def test_omega_zero(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2) * 1.0
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2.0

        mean_fused, cov_fused = covariance_intersection(
            mean_a, cov_a, mean_b, cov_b, omega=0.0
        )
        # omega=0 means all weight on B
        np.testing.assert_array_almost_equal(mean_fused, mean_b)

    def test_omega_one(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2) * 1.0
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2.0

        mean_fused, cov_fused = covariance_intersection(
            mean_a, cov_a, mean_b, cov_b, omega=1.0
        )
        # omega=1 means all weight on A
        np.testing.assert_array_almost_equal(mean_fused, mean_a)

    def test_optimal_omega(self):
        mean_a = np.array([[1.0], [2.0]])
        cov_a = np.eye(2) * 1.0
        mean_b = np.array([[3.0], [4.0]])
        cov_b = np.eye(2) * 2.0

        mean_fused, cov_fused = covariance_intersection(
            mean_a, cov_a, mean_b, cov_b, omega=None
        )
        assert mean_fused.shape == (2, 1)

    def test_sequential_fusion(self):
        estimates = [
            (np.array([[1.0], [2.0]]), np.eye(2) * 1.0),
            (np.array([[3.0], [4.0]]), np.eye(2) * 2.0),
            (np.array([[5.0], [6.0]]), np.eye(2) * 3.0),
        ]

        mean_fused, cov_fused = sequential_covariance_intersection(estimates)
        assert mean_fused.shape == (2, 1)
        assert cov_fused.shape == (2, 2)

    def test_sequential_single(self):
        estimates = [
            (np.array([[1.0], [2.0]]), np.eye(2) * 1.0),
        ]

        mean_fused, cov_fused = sequential_covariance_intersection(estimates)
        np.testing.assert_array_almost_equal(mean_fused, estimates[0][0])

    def test_sequential_empty(self):
        with pytest.raises(ValueError):
            sequential_covariance_intersection([])


class TestProceduralAPI:
    """Tests for procedural filter functions."""

    def test_kf_predict(self):
        x = np.array([[0.0], [1.0]])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01

        x_pred, P_pred = kf_predict(x, P, F, Q)
        assert x_pred.shape == (2, 1)
        assert P_pred.shape == (2, 2)
        np.testing.assert_array_almost_equal(x_pred, [[1.0], [1.0]])

    def test_kf_predict_with_control(self):
        x = np.array([[0.0], [1.0]])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        B = np.array([[0.5], [1.0]])
        u = np.array([[0.1]])

        x_pred, P_pred = kf_predict(x, P, F, Q, B=B, u=u)
        np.testing.assert_array_almost_equal(x_pred, [[1.05], [1.1]])

    def test_kf_update(self):
        x = np.array([[1.0], [1.0]])
        P = np.eye(2)
        z = np.array([[1.2]])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        x_upd, P_upd, y, S = kf_update(x, P, z, H, R)
        assert x_upd.shape == (2, 1)
        assert P_upd.shape == (2, 2)
        np.testing.assert_array_almost_equal(y, [[0.2]])

    def test_compute_kalman_gain(self):
        P = np.eye(2)
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        K = compute_kalman_gain(P, H, R)
        assert K.shape == (2, 1)

    def test_compute_nis(self):
        innovation = np.array([[0.5]])
        S = np.array([[1.0]])

        nis_val = compute_nis(innovation, S)
        assert nis_val == pytest.approx(0.25)

    def test_compute_nees(self):
        state_error = np.array([[0.1], [0.05]])
        P = np.eye(2) * 0.1

        nees_val = compute_nees(state_error, P)
        assert nees_val == pytest.approx(0.125)

    def test_predict_update_consistency(self):
        x = np.array([[0.0], [1.0]])
        P = np.eye(2) * 10.0
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        for z in measurements:
            x, P = kf_predict(x, P, F, Q)
            x, P, _, _ = kf_update(x, P, np.array([[z]]), H, R)

        assert x[0, 0] > 0
        assert P[0, 0] > 0
