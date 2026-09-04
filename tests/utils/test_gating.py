"""Tests for innovation gating."""

import numpy as np
import pytest

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.utils.gating import (
    nis,
    chi2_gate,
    mahalanobis_distance,
    ellipsoidal_gate,
    gated_update,
)


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
