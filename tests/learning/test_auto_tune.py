"""Tests for NIS-based auto-tuning of Q/R."""

import numpy as np

from kalbee.modules.learning.auto_tune import (
    tune_kalman_filter,
    quick_tune,
    TuneResult,
)


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
            measurements,
            F,
            H,
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
