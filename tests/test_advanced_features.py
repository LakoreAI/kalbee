"""Tests for H-infinity filter, consistency tests, auto-tuning, and sigma points."""
import numpy as np
import pytest

from kalbee.modules.filters.hinfinity_filter import HInfinityFilter
from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.sigma_points import (
    SimplexSigmaPoints,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)
from kalbee.modules.utils.consistency import (
    nis_test,
    nees_test,
    innovation_whiteness_test,
)
from kalbee.modules.learning.auto_tune import (
    tune_kalman_filter,
    quick_tune,
    TuneResult,
)


# ============================================================
# H-Infinity Filter tests
# ============================================================

class TestHInfinityFilter:
    def test_initialization(self):
        kf = HInfinityFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
        )
        assert kf.gamma == 10.0

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            HInfinityFilter(
                state=np.array([[0.0]]),
                covariance=np.eye(1),
                transition_matrix=np.eye(1),
                process_noise_cov=np.eye(1) * 0.01,
                measurement_matrix=np.eye(1),
                measurement_noise_cov=np.array([[0.1]]),
                gamma=-1.0,
            )

    def test_predict_update_cycle(self):
        kf = HInfinityFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
        )
        for _ in range(10):
            kf.predict()
            kf.update(np.array([[1.0]]))
        assert np.abs(kf.state[0, 0] - 1.0) < 2.0

    def test_high_gamma_approaches_kf(self):
        """With large gamma, H-infinity should behave like standard KF."""
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2) * 10.0
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        kf_std = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)
        kf_hinf = HInfinityFilter(state.copy(), cov.copy(), F, Q, H, R, gamma=1000.0)

        kf_std.predict()
        kf_hinf.predict()
        kf_std.update(np.array([[1.0]]))
        kf_hinf.update(np.array([[1.0]]))

        # Should be very close
        assert np.allclose(kf_std.state, kf_hinf.state, atol=0.1)

    def test_with_control_input(self):
        kf = HInfinityFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            control_matrix=np.array([[0.5], [0.1]]),
        )
        kf.predict(u=np.array([[1.0]]))
        assert kf.state[0, 0] != 0.0


# ============================================================
# Sigma Points tests
# ============================================================

class TestSimplexSigmaPoints:
    def test_sigma_points_shape(self):
        sp = SimplexSigmaPoints(n=2, alpha=0.001, beta=2.0, kappa=0.0)
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        sigmas = sp.sigma_points(x, P)
        assert sigmas.shape == (5, 2)

    def test_sigma_points_mean(self):
        sp = SimplexSigmaPoints(n=2, alpha=0.001, beta=2.0, kappa=0.0)
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        sigmas = sp.sigma_points(x, P)
        # Weighted mean should recover the original mean
        mean = np.dot(sp.weights_mean, sigmas)
        assert np.allclose(mean, x, atol=1e-10)

    def test_num_sigma_points(self):
        sp = SimplexSigmaPoints(n=3)
        assert sp.num_sigma_points == 7


class TestMerweScaledSigmaPoints:
    def test_sigma_points_shape(self):
        sp = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=0.0)
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3)
        sigmas = sp.sigma_points(x, P)
        assert sigmas.shape == (7, 3)

    def test_weights_sum_to_one(self):
        sp = MerweScaledSigmaPoints(n=2)
        assert np.sum(sp.weights_mean) == pytest.approx(1.0)


class TestJulierSigmaPoints:
    def test_sigma_points_shape(self):
        sp = JulierSigmaPoints(n=2, kappa=0.0)
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        sigmas = sp.sigma_points(x, P)
        assert sigmas.shape == (5, 2)

    def test_weights_symmetry(self):
        sp = JulierSigmaPoints(n=2, kappa=0.0)
        # Julier weights should be symmetric (except for the first)
        assert sp.weights_mean[1] == pytest.approx(sp.weights_mean[-1])


# ============================================================
# Consistency Tests
# ============================================================

class TestNISTest:
    def test_consistent_innovations(self):
        """White Gaussian innovations should pass the NIS test."""
        np.random.seed(42)
        m = 2
        T = 100
        innovations = [np.random.randn(m, 1) for _ in range(T)]
        # Covariance = I, so NIS = v'v, which is chi-squared(m)
        innovation_covs = [np.eye(m) for _ in range(T)]

        passed, nis_vals, mean_nis, expected, p_val = nis_test(innovations, innovation_covs)
        assert passed
        assert mean_nis == pytest.approx(m, abs=0.5)

    def test_inconsistent_innovations(self):
        """Biased innovations should fail the NIS test."""
        np.random.seed(42)
        m = 1
        T = 100
        # Add large bias
        innovations = [np.array([[5.0 + np.random.randn()]]) for _ in range(T)]
        innovation_covs = [np.eye(m) for _ in range(T)]

        passed, _, mean_nis, _, _ = nis_test(innovations, innovation_covs)
        # Mean NIS should be much larger than m=1
        assert mean_nis > 10.0


class TestNEESTest:
    def test_consistent_errors(self):
        """Errors matching the covariance should pass."""
        np.random.seed(42)
        n = 2
        T = 100
        state_errors = [np.random.randn(n, 1) * 0.1 for _ in range(T)]
        covariances = [np.eye(n) * 0.01 for _ in range(T)]

        passed, nees_vals, mean_nees, expected, p_val = nees_test(state_errors, covariances)
        # Mean NEES should be close to n=2
        assert passed


class TestInnovationWhitenessTest:
    def test_white_innovations(self):
        """Random innovations should be white."""
        np.random.seed(42)
        T = 200
        innovations = [np.array([[np.random.randn()]]) for _ in range(T)]

        passed, autocorr = innovation_whiteness_test(innovations, max_lag=10)
        assert passed

    def test_correlated_innovations(self):
        """Correlated innovations should fail the whiteness test."""
        np.random.seed(42)
        T = 200
        innovations = []
        val = 0.0
        for _ in range(T):
            val = 0.9 * val + np.random.randn()  # AR(1) process
            innovations.append(np.array([[val]]))

        passed, autocorr = innovation_whiteness_test(innovations, max_lag=10)
        assert not passed


# ============================================================
# Auto-tuning tests
# ============================================================

class TestAutoTune:
    def test_tune_kalman_filter(self):
        np.random.seed(42)
        T = 200
        dt = 1.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        true_Q = np.array([[0.01, 0.0], [0.0, 0.01]])
        true_R = np.array([[0.1]])

        # Generate data
        x = np.zeros((2, 1))
        measurements = []
        for _ in range(T):
            x = F @ x + np.random.multivariate_normal(np.zeros(2), true_Q).reshape(2, 1)
            z = H @ x + np.random.multivariate_normal(np.zeros(1), true_R).reshape(1, 1)
            measurements.append(z.flatten())
        measurements = np.array(measurements)

        result = tune_kalman_filter(
            measurements, F, H,
            Q_init=np.eye(2) * 0.001,
            R_init=np.eye(1) * 0.01,
            n_iter=20,
        )

        assert isinstance(result, TuneResult)
        assert result.Q.shape == (2, 2)
        assert result.R.shape == (1, 1)
        assert len(result.nis_history) > 0

    def test_quick_tune(self):
        np.random.seed(42)
        T = 100
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        measurements = np.random.randn(T, 1) + 5.0

        Q, R = quick_tune(measurements, F, H)
        assert Q.shape == (2, 2)
        assert R.shape == (1, 1)
        # R should be adjusted to make NIS reasonable
        assert R[0, 0] > 0


# ============================================================
# AutoFilter H-infinity test
# ============================================================

class TestAutoFilterHInfinity:
    def test_hinf_short(self):
        from kalbee.modules.filters.auto_filter import AutoFilter
        kf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            mode="hinf",
        )
        assert isinstance(kf, HInfinityFilter)

    def test_hinf_full_name(self):
        from kalbee.modules.filters.auto_filter import AutoFilter
        kf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            mode="HInfinityFilter",
        )
        assert isinstance(kf, HInfinityFilter)

    def test_hinf_alias(self):
        from kalbee.modules.filters.auto_filter import AutoFilter
        kf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            mode="hinfinity",
        )
        assert isinstance(kf, HInfinityFilter)
