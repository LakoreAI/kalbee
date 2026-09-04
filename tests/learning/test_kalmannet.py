"""Tests for the KalmanNet hybrid neural filter (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from kalbee.modules.learning.kalmannet import KalmanNet  # noqa: E402


def test_kalmannet():
    """KalmanNet should run PyTorch step when torch is available."""
    knet = KalmanNet(state_dim=2, meas_dim=1, hidden_dim=16)
    x_pred = torch.zeros(1, 2, 1)
    z = torch.ones(1, 1, 1)
    H = torch.tensor([[1.0, 0.0]]).unsqueeze(0)
    h_rnn = torch.zeros(1, 16)

    x_upd, h_next = knet.step(x_pred, z, H, h_rnn)
    assert x_upd.shape == (1, 2, 1)
    assert h_next.shape == (1, 16)
