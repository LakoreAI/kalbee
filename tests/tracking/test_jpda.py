import numpy as np

from kalbee import JPDAAssociation


def test_jpda_association():
    """JPDAAssociation should return marginal association probabilities matrix beta."""
    jpda = JPDAAssociation(p_d=0.9, clutter_density=1e-3, gate_threshold=16.0)

    track_states = [
        np.array([[0.0], [0.0]]),
        np.array([[10.0], [0.0]]),
    ]
    track_covs = [np.eye(2) * 1.0, np.eye(2) * 1.0]

    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])
    measurements = np.array([[0.1], [9.9], [50.0]])  # 3 detections

    beta = jpda.compute_association_probabilities(
        track_states, track_covs, H, R, measurements
    )

    assert beta.shape == (2, 4)  # 2 tracks x (3 detections + 1 missed)
    np.testing.assert_allclose(beta.sum(axis=1), np.ones(2), atol=1e-6)
    # Track 0 should have high probability for measurement 0
    assert beta[0, 1] > beta[0, 2]
    # Track 1 should have high probability for measurement 1
    assert beta[1, 2] > beta[1, 1]
