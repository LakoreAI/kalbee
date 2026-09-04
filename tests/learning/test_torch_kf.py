"""Tests for the differentiable PyTorch Kalman filter (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from kalbee.modules.learning.torch_kf import DifferentiableKalmanFilter  # noqa: E402


def test_differentiable_kalman_filter():
    """DifferentiableKalmanFilter should run a predict/update step."""
    dkf = DifferentiableKalmanFilter(state_dim=2, meas_dim=1)
    x = torch.zeros(1, 2, 1)
    cov = torch.eye(2).unsqueeze(0)
    z = torch.ones(1, 1, 1)

    x_pred, cov_pred = dkf.predict(x, cov)
    x_upd, cov_upd = dkf.update(x_pred, cov_pred, z)

    assert x_upd.shape == (1, 2, 1)
