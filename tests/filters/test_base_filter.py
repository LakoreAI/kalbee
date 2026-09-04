"""Tests for BaseFilter shared behavior (predict_only, reset, batch, serialization)."""

import os
import tempfile

import numpy as np
import pytest

from kalbee.modules.filters.kf_filter import KalmanFilter


def _kf2():
    return KalmanFilter(
        state=np.array([[0.0], [0.0]]),
        covariance=np.eye(2),
        transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
        transition_covariance=np.eye(2) * 0.01,
        measurement_matrix=np.array([[1.0, 0.0]]),
        measurement_covariance=np.array([[0.1]]),
    )


# ============================================================
# predict_only
# ============================================================


class TestBaseFilterPredictOnly:
    def test_predict_only_does_not_modify_state(self):
        kf = _kf2()
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


# ============================================================
# reset
# ============================================================


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
# batch processing (filter_sequence)
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
        kf = _kf2()
        measurements = np.random.randn(5)
        state_hist, _ = kf.filter_sequence(measurements)
        assert state_hist.shape == (5, 2, 1)


# ============================================================
# serialization (save_state / load_state)
# ============================================================


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
            kf2 = _kf2()
            kf2.state = np.zeros((2, 1))
            kf2.covariance = np.eye(2)
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
