import numpy as np

from kalbee import VariationalBayesKalmanFilter


def test_variational_bayes_kalman_filter():
    """VBAKF should estimate state and adjust measurement covariance R online."""
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2) * 10.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R_init = np.array([[1.0]])

    vbakf = VariationalBayesKalmanFilter(state, cov, F, Q, H, R_init, n_iter=3)

    for t in range(5):
        vbakf.predict()
        vbakf.update(np.array([[float(t + 1)]]))

    assert vbakf.state.shape == (2, 1)
    assert vbakf.measurement_covariance.shape == (1, 1)
