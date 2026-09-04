import numpy as np
import pytest

from kalbee import FadingMemoryKalmanFilter, KalmanFilter


class TestFadingMemoryKalmanFilter:
    def test_initialization(self):
        kf = FadingMemoryKalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            fading_factor=1.05,
        )
        assert kf.fading_factor == 1.05

    def test_invalid_fading_factor(self):
        with pytest.raises(ValueError, match="Fading factor must be >= 1.0"):
            FadingMemoryKalmanFilter(
                state=np.array([[0.0], [0.0]]),
                covariance=np.eye(2),
                transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
                transition_covariance=np.eye(2) * 0.01,
                measurement_matrix=np.array([[1.0, 0.0]]),
                measurement_covariance=np.array([[0.1]]),
                fading_factor=0.9,
            )

    def test_predict_inflates_covariance(self):
        kf_std = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        kf_fm = FadingMemoryKalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            fading_factor=1.1,
        )
        kf_std.predict()
        kf_fm.predict()
        # Fading memory should have larger covariance
        assert np.trace(kf_fm.covariance) > np.trace(kf_std.covariance)

    def test_alpha_1_equals_standard_kf(self):
        """With fading_factor=1.0, should behave identically to standard KF."""
        kf_std = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        kf_fm = FadingMemoryKalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            fading_factor=1.0,
        )
        kf_std.predict()
        kf_fm.predict()
        assert np.allclose(kf_std.covariance, kf_fm.covariance)
        assert np.allclose(kf_std.state, kf_fm.state)

    def test_predict_update_cycle(self):
        kf = FadingMemoryKalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            fading_factor=1.05,
        )
        for _ in range(10):
            kf.predict()
            kf.update(np.array([[1.0]]))
        assert np.abs(kf.state[0, 0] - 1.0) < 2.0

    def test_with_control_input(self):
        kf = FadingMemoryKalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            control_matrix=np.array([[0.5], [0.1]]),
            fading_factor=1.05,
        )
        kf.predict(u=np.array([[1.0]]))
        # State should be affected by control input
        assert kf.state[0, 0] != 0.0
