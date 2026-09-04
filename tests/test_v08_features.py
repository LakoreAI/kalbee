"""Tests for SigmaPointUKF, FilterDiagnostics, Chi2OutlierDetector, predict_only, and reset."""
import numpy as np
import pytest

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.sigma_point_ukf import SigmaPointUKF
from kalbee.modules.filters.sigma_points import (
    SimplexSigmaPoints,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)
from kalbee.modules.utils.diagnostics import FilterDiagnostics
from kalbee.modules.utils.outlier_detector import Chi2OutlierDetector


# ============================================================
# SigmaPointUKF tests
# ============================================================

class TestSigmaPointUKF:
    def test_initialization_default_sigma(self):
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
        )
        assert isinstance(ukf.sigma_points, SimplexSigmaPoints)

    def test_initialization_custom_sigma(self):
        sp = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0.0)
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            sigma_points=sp,
        )
        assert isinstance(ukf.sigma_points, MerweScaledSigmaPoints)

    def test_predict_update_cycle(self):
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: np.array([[x[0, 0] + x[1, 0] * dt], [x[1, 0]]]),
            measurement_function=lambda x: x[:1],
        )
        for _ in range(10):
            ukf.predict()
            ukf.update(np.array([[1.0]]))
        assert np.abs(ukf.state[0, 0] - 1.0) < 2.0

    def test_julier_sigma_points(self):
        sp = JulierSigmaPoints(n=2, kappa=0.0)
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            sigma_points=sp,
        )
        ukf.predict()
        assert ukf.state.shape == (2, 1)


# ============================================================
# FilterDiagnostics tests
# ============================================================

class TestFilterDiagnostics:
    def test_collect(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        diag = FilterDiagnostics(m=1, n=2)

        kf.predict()
        kf.update(np.array([[1.0]]))
        snapshot = diag.collect(kf, np.array([[1.0]]))

        assert snapshot.timestamp == 0
        assert len(diag.nis_history) == 1
        assert len(diag.state_history) == 1

    def test_summary(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        diag = FilterDiagnostics(m=1, n=2)

        for _ in range(20):
            kf.predict()
            kf.update(np.array([[1.0]]))
            diag.collect(kf, np.array([[1.0]]))

        report = diag.summary()
        assert report["num_steps"] == 20
        assert "nis_mean" in report
        assert "cov_trace_final" in report

    def test_check_consistency(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        diag = FilterDiagnostics(m=1, n=2)

        for _ in range(20):
            kf.predict()
            kf.update(np.array([[1.0]]))
            diag.collect(kf, np.array([[1.0]]))

        results = diag.check_consistency()
        assert "nis_consistent" in results

    def test_reset(self):
        diag = FilterDiagnostics(m=1, n=2)
        diag.nis_history = [1.0, 2.0, 3.0]
        diag.reset()
        assert len(diag.nis_history) == 0

    def test_get_innovations(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        diag = FilterDiagnostics(m=1, n=2)

        for _ in range(5):
            kf.predict()
            kf.update(np.array([[1.0]]))
            diag.collect(kf, np.array([[1.0]]))

        innovations = diag.get_innovations()
        assert innovations.shape == (5, 1)


# ============================================================
# Chi2OutlierDetector tests
# ============================================================

class TestChi2OutlierDetector:
    def test_inlier_detection(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95)
        innovation = np.array([[0.1]])
        innovation_cov = np.array([[1.0]])
        result = det.check(innovation, innovation_cov)
        assert result.is_inlier

    def test_outlier_detection(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95)
        innovation = np.array([[10.0]])
        innovation_cov = np.array([[0.1]])
        result = det.check(innovation, innovation_cov)
        assert not result.is_inlier

    def test_adaptive_mode(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95, adaptive=True)
        # Feed many normal measurements
        for _ in range(50):
            det.check(np.array([[0.1]]), np.array([[1.0]]))
        # Check that adaptive threshold has been updated
        assert det._adaptive_threshold != det.fixed_threshold

    def test_batch_check(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95)
        innovations = np.array([[0.1], [0.2], [10.0]])
        innovation_covs = np.array([[[1.0]], [[1.0]], [[0.1]]])
        results = det.batch_check(innovations, innovation_covs)
        assert len(results) == 3
        assert results[0].is_inlier
        assert not results[2].is_inlier

    def test_statistics(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95)
        for _ in range(10):
            det.check(np.array([[0.1]]), np.array([[1.0]]))
        stats = det.get_statistics()
        assert stats["num_checks"] == 10
        assert "nis_mean" in stats

    def test_statistics_adaptive(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95, adaptive=True)
        for _ in range(20):
            det.check(np.array([[0.1]]), np.array([[1.0]]))
        stats = det.get_statistics()
        assert stats["num_checks"] == 20
        assert "current_threshold" in stats

    def test_reset(self):
        det = Chi2OutlierDetector(m=1, confidence=0.95, adaptive=True)
        for _ in range(10):
            det.check(np.array([[0.1]]), np.array([[1.0]]))
        det.reset()
        assert len(det._nis_buffer) == 0


# ============================================================
# BaseFilter predict_only and reset tests
# ============================================================

class TestBaseFilterPredictOnly:
    def test_predict_only_does_not_modify_state(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        state_before = kf.state.copy()
        predicted = kf.predict_only()
        assert np.allclose(kf.state, state_before)
        assert predicted[0, 0] == pytest.approx(0.0)  # Predicted position

    def test_predict_only_returns_correct_prediction(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [1.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        predicted = kf.predict_only()
        # x = F @ x = [[1,1],[0,1]] @ [0,1] = [1, 1]
        assert predicted[0, 0] == pytest.approx(1.0)
        assert predicted[1, 0] == pytest.approx(1.0)


class TestBaseFilterReset:
    def test_reset_to_zeros(self):
        kf = KalmanFilter(
            state=np.array([[5.0], [3.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.eye(2),
            transition_covariance=np.eye(2),
            measurement_matrix=np.eye(2),
            measurement_covariance=np.eye(2),
        )
        kf.reset()
        assert np.allclose(kf.state, np.zeros((2, 1)))
        assert np.allclose(kf.covariance, np.eye(2) * 100.0)

    def test_reset_to_specific_state(self):
        kf = KalmanFilter(
            state=np.array([[5.0], [3.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.eye(2),
            transition_covariance=np.eye(2),
            measurement_matrix=np.eye(2),
            measurement_covariance=np.eye(2),
        )
        new_state = np.array([[1.0], [2.0]])
        new_cov = np.eye(2) * 5.0
        kf.reset(state=new_state, covariance=new_cov)
        assert np.allclose(kf.state, new_state)
        assert np.allclose(kf.covariance, new_cov)


# ============================================================
# AutoFilter SigmaPointUKF test
# ============================================================

class TestAutoFilterSPUKF:
    def test_spukf_short(self):
        from kalbee.modules.filters.auto_filter import AutoFilter
        ukf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            mode="spukf",
        )
        assert isinstance(ukf, SigmaPointUKF)

    def test_spukf_full_name(self):
        from kalbee.modules.filters.auto_filter import AutoFilter
        ukf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            mode="SigmaPointUKF",
        )
        assert isinstance(ukf, SigmaPointUKF)
