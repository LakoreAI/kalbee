import numpy as np
import pytest

from kalbee import covariance_intersection, sequential_covariance_intersection


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
