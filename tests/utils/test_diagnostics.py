"""Tests for real-time FilterDiagnostics."""

import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.utils.diagnostics import FilterDiagnostics


def _kf():
    return KalmanFilter(
        state=np.array([[0.0], [0.0]]),
        covariance=np.eye(2) * 10.0,
        transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
        transition_covariance=np.eye(2) * 0.01,
        measurement_matrix=np.array([[1.0, 0.0]]),
        measurement_covariance=np.array([[0.1]]),
    )


class TestFilterDiagnostics:
    def test_collect(self):
        kf = _kf()
        diag = FilterDiagnostics(m=1, n=2)

        kf.predict()
        kf.update(np.array([[1.0]]))
        snapshot = diag.collect(kf, np.array([[1.0]]))

        assert snapshot.timestamp == 0
        assert len(diag.nis_history) == 1
        assert len(diag.state_history) == 1

    def test_summary(self):
        kf = _kf()
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
        kf = _kf()
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
        kf = _kf()
        diag = FilterDiagnostics(m=1, n=2)

        for _ in range(5):
            kf.predict()
            kf.update(np.array([[1.0]]))
            diag.collect(kf, np.array([[1.0]]))

        innovations = diag.get_innovations()
        assert innovations.shape == (5, 1)
