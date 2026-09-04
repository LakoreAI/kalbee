import numpy as np

from kalbee import OnlineEM


class TestOnlineEM:
    """Tests for Online EM."""

    def test_basic_update(self):
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])

        em = OnlineEM(F, H, forgetting_factor=0.99)

        state = np.array([[1.0], [0.5]])
        cov = np.eye(2) * 0.1
        pred = np.array([[1.5], [0.5]])
        pred_cov = np.eye(2) * 0.2
        z = np.array([[1.2]])

        Q, R = em.update(state, cov, pred, pred_cov, z)

        assert Q.shape == (2, 2)
        assert R.shape == (1, 1)
        assert em.sample_count == 1
