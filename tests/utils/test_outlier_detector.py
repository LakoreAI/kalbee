"""Tests for the real-time Chi2OutlierDetector."""

import numpy as np

from kalbee.modules.utils.outlier_detector import Chi2OutlierDetector


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
