"""Tests for innovation gating, fading memory filter, and base filter utilities."""
import numpy as np
import pytest
import tempfile
import os

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.fading_memory_kf import FadingMemoryKalmanFilter
from kalbee.modules.utils.gating import (
    nis,
    chi2_gate,
    mahalanobis_distance,
    ellipsoidal_gate,
    gated_update,
)


# ============================================================
# Gating tests
# ============================================================

class TestNIS:
    def test_perfect_innovation(self):
        """Zero innovation should give NIS = 0."""
        v = np.array([[0.0], [0.0]])
        S = np.eye(2)
        assert nis(v, S) == pytest.approx(0.0)

    def test_nonzero_innovation(self):
        v = np.array([[1.0], [0.0]])
        S = np.eye(2)
        assert nis(v, S) == pytest.approx(1.0)

    def test_scaled_innovation(self):
        v = np.array([[2.0], [0.0]])
        S = np.array([[4.0, 0.0], [0.0, 1.0]])
        # NIS = v' S^-1 v = [2, 0] @ [[0.25, 0], [0, 1]] @ [2, 0] = 1.0
        assert nis(v, S) == pytest.approx(1.0)


class TestChi2Gate:
    def test_passes_for_small_innovation(self):
        v = np.array([[0.1], [0.1]])
        S = np.eye(2)
        passed, nis_val, threshold = chi2_gate(v, S, confidence=0.95)
        assert passed is True
        assert nis_val < threshold

    def test_fails_for_large_innovation(self):
        v = np.array([[10.0], [10.0]])
        S = np.eye(2) * 0.1
        passed, nis_val, threshold = chi2_gate(v, S, confidence=0.95)
        assert passed is False
        assert nis_val > threshold

    def test_threshold_increases_with_confidence(self):
        v = np.array([[1.0]])
        S = np.array([[1.0]])
        _, _, t90 = chi2_gate(v, S, confidence=0.90)
        _, _, t99 = chi2_gate(v, S, confidence=0.99)
        assert t99 > t90

    def test_1d_measurement(self):
        v = np.array([[0.5]])
        S = np.array([[1.0]])
        passed, _, _ = chi2_gate(v, S, confidence=0.95)
        assert passed is True


class TestMahalanobisDistance:
    def test_zero_distance(self):
        v = np.array([[0.0], [0.0]])
        S = np.eye(2)
        assert mahalanobis_distance(v, S) == pytest.approx(0.0)

    def test_unit_distance(self):
        v = np.array([[1.0], [0.0]])
        S = np.eye(2)
        assert mahalanobis_distance(v, S) == pytest.approx(1.0)

    def test_scaled_distance(self):
        v = np.array([[2.0], [0.0]])
        S = np.array([[4.0, 0.0], [0.0, 1.0]])
        assert mahalanobis_distance(v, S) == pytest.approx(1.0)


class TestEllipsoidalGate:
    def test_passes_within_gate(self):
        v = np.array([[0.5], [0.5]])
        S = np.eye(2)
        assert ellipsoidal_gate(v, S, gate_threshold=5.0) is True

    def test_fails_outside_gate(self):
        v = np.array([[10.0], [10.0]])
        S = np.eye(2)
        assert ellipsoidal_gate(v, S, gate_threshold=5.0) is False


class TestGatedUpdate:
    def test_accepted_measurement(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        kf.predict()
        updated, state = gated_update(kf, np.array([[0.5]]), confidence=0.95)
        assert updated is True

    def test_rejected_outlier(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        kf.predict()
        state_before = kf.state.copy()
        updated, state = gated_update(kf, np.array([[100.0]]), confidence=0.95)
        assert updated is False
        assert np.allclose(kf.state, state_before)

    def test_mahalanobis_gating(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        kf.predict()
        updated, _ = gated_update(kf, np.array([[0.5]]), gate_threshold=5.0)
        assert updated is True


# ============================================================
# Fading Memory Filter tests
# ============================================================

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


# ============================================================
# BaseFilter batch processing and serialization tests
# ============================================================

class TestBaseFilterBatchProcessing:
    def test_filter_sequence(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        measurements = np.random.randn(20, 1) + 5.0
        state_hist, cov_hist = kf.filter_sequence(measurements)
        assert state_hist.shape == (20, 2, 1)
        assert len(cov_hist) == 20

    def test_filter_sequence_with_missing(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        measurements = np.random.randn(10, 1)
        measurements[3] = np.nan  # missing measurement
        measurements[7] = np.nan
        state_hist, cov_hist = kf.filter_sequence(measurements, missing=np.nan)
        assert state_hist.shape == (10, 2, 1)

    def test_filter_sequence_1d(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        measurements = np.random.randn(5)
        state_hist, _ = kf.filter_sequence(measurements)
        assert state_hist.shape == (5, 2, 1)


class TestBaseFilterSerialization:
    def test_save_and_load_state(self):
        kf = KalmanFilter(
            state=np.array([[1.0], [2.0]]),
            covariance=np.eye(2) * 5.0,
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        kf.predict()
        kf.update(np.array([[1.5]]))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            kf.save_state(filepath)
            kf2 = KalmanFilter(
                state=np.zeros((2, 1)),
                covariance=np.eye(2),
                transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
                transition_covariance=np.eye(2) * 0.01,
                measurement_matrix=np.array([[1.0, 0.0]]),
                measurement_covariance=np.array([[0.1]]),
            )
            kf2.load_state(filepath)
            assert np.allclose(kf.state, kf2.state)
            assert np.allclose(kf.covariance, kf2.covariance)
        finally:
            os.unlink(filepath)

    def test_save_state_minimal(self):
        """Test saving when matrices are None."""
        kf = KalmanFilter(
            state=np.array([[0.0]]),
            covariance=np.eye(1),
            transition_matrix=None,
            transition_covariance=None,
            measurement_matrix=None,
            measurement_covariance=None,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            kf.save_state(filepath)
            kf2 = KalmanFilter(
                state=np.zeros((1, 1)),
                covariance=np.eye(1),
                transition_matrix=None,
                transition_covariance=None,
                measurement_matrix=None,
                measurement_covariance=None,
            )
            kf2.load_state(filepath)
            assert np.allclose(kf.state, kf2.state)
        finally:
            os.unlink(filepath)


# ============================================================
# KF control input tests
# ============================================================

class TestKFControlInput:
    def test_predict_with_control(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            control_matrix=np.array([[0.5], [0.1]]),
        )
        kf.predict(u=np.array([[2.0]]))
        # x = F @ x + B @ u = [[0,1],[0,0]] @ [0,0] + [[0.5],[0.1]] @ [2] = [1.0, 0.2]
        assert kf.state[0, 0] == pytest.approx(1.0)
        assert kf.state[1, 0] == pytest.approx(0.2)

    def test_predict_without_control(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
            control_matrix=np.array([[0.5], [0.1]]),
        )
        kf.predict()
        # x = F @ x = [[1,1],[0,1]] @ [0,0] = [0, 0]
        assert kf.state[0, 0] == pytest.approx(0.0)
        assert kf.state[1, 0] == pytest.approx(0.0)

    def test_predict_with_B_in_kwargs(self):
        kf = KalmanFilter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            transition_covariance=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_covariance=np.array([[0.1]]),
        )
        B = np.array([[0.5], [0.1]])
        kf.predict(u=np.array([[2.0]]), B=B)
        assert kf.state[0, 0] == pytest.approx(1.0)
