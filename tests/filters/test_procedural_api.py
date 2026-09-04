"""Tests for the procedural (functional) filter API."""

import numpy as np
import pytest

from kalbee import (
    kf_predict,
    kf_update,
    compute_kalman_gain,
    compute_nis,
    compute_nees,
)


class TestProceduralAPI:
    """Tests for procedural filter functions."""

    def test_kf_predict(self):
        x = np.array([[0.0], [1.0]])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01

        x_pred, P_pred = kf_predict(x, P, F, Q)
        assert x_pred.shape == (2, 1)
        assert P_pred.shape == (2, 2)
        np.testing.assert_array_almost_equal(x_pred, [[1.0], [1.0]])

    def test_kf_predict_with_control(self):
        x = np.array([[0.0], [1.0]])
        P = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        B = np.array([[0.5], [1.0]])
        u = np.array([[0.1]])

        x_pred, P_pred = kf_predict(x, P, F, Q, B=B, u=u)
        np.testing.assert_array_almost_equal(x_pred, [[1.05], [1.1]])

    def test_kf_update(self):
        x = np.array([[1.0], [1.0]])
        P = np.eye(2)
        z = np.array([[1.2]])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        x_upd, P_upd, y, S = kf_update(x, P, z, H, R)
        assert x_upd.shape == (2, 1)
        assert P_upd.shape == (2, 2)
        np.testing.assert_array_almost_equal(y, [[0.2]])

    def test_compute_kalman_gain(self):
        P = np.eye(2)
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        K = compute_kalman_gain(P, H, R)
        assert K.shape == (2, 1)

    def test_compute_nis(self):
        innovation = np.array([[0.5]])
        S = np.array([[1.0]])

        nis_val = compute_nis(innovation, S)
        assert nis_val == pytest.approx(0.25)

    def test_compute_nees(self):
        state_error = np.array([[0.1], [0.05]])
        P = np.eye(2) * 0.1

        nees_val = compute_nees(state_error, P)
        assert nees_val == pytest.approx(0.125)

    def test_predict_update_consistency(self):
        x = np.array([[0.0], [1.0]])
        P = np.eye(2) * 10.0
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]
        for z in measurements:
            x, P = kf_predict(x, P, F, Q)
            x, P, _, _ = kf_update(x, P, np.array([[z]]), H, R)

        assert x[0, 0] > 0
        assert P[0, 0] > 0
