"""Tests for NIS/NEES/whiteness consistency tests."""

import numpy as np
import pytest

from kalbee.modules.utils.consistency import (
    nis_test,
    nees_test,
    innovation_whiteness_test,
)


class TestNISTest:
    def test_consistent_innovations(self):
        """White Gaussian innovations should pass the NIS test."""
        np.random.seed(42)
        m = 2
        T = 100
        innovations = [np.random.randn(m, 1) for _ in range(T)]
        # Covariance = I, so NIS = v'v, which is chi-squared(m)
        innovation_covs = [np.eye(m) for _ in range(T)]

        passed, nis_vals, mean_nis, expected, p_val = nis_test(
            innovations, innovation_covs
        )
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

        passed, nees_vals, mean_nees, expected, p_val = nees_test(
            state_errors, covariances
        )
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
