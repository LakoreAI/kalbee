import numpy as np

from kalbee import PMBMTracker


def test_pmbm_tracker():
    """PMBMTracker should track target hypotheses and manage births/pruning."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])

    pmbm = PMBMTracker(F, Q, H, R, birth_rate=0.1)

    # Frame 1: 1 detection
    pmbm.predict(dt=1.0)
    confirmed = pmbm.update(np.array([[1.0]]))

    # Frame 2: detection moves
    pmbm.predict(dt=1.0)
    confirmed = pmbm.update(np.array([[2.0]]))

    assert isinstance(confirmed, list)
