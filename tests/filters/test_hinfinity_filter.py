"""Tests for the H-Infinity filter."""

import numpy as np
import pytest

from kalbee.modules.filters.hinfinity_filter import HInfinityFilter
from kalbee.modules.filters.kf_filter import KalmanFilter


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

    def test_robust_covariance_predict_update(self):
        """H-Infinity filter predict/update cycle with a mid-range gamma."""
        kf = HInfinityFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=5.0,
        )
        kf.predict()
        state = kf.update(np.array([[1.0]]))
        assert state.shape == (2, 1)
