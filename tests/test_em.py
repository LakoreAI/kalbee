import numpy as np

from kalbee import em_kalman
from kalbee.models import constant_velocity, position_measurement_model


def _simulate(F, H, Q, R, x0, T, rng):
    n = F.shape[0]
    m = H.shape[0]
    x = x0.copy()
    zs = np.zeros((T, m))
    Lq = np.linalg.cholesky(Q + np.eye(n) * 1e-12)
    Lr = np.linalg.cholesky(R + np.eye(m) * 1e-12)
    for t in range(T):
        x = F @ x + Lq @ rng.standard_normal((n, 1))
        z = H @ x + Lr @ rng.standard_normal((m, 1))
        zs[t] = z.ravel()
    return zs


def test_em_loglik_monotonic_nondecreasing():
    rng = np.random.default_rng(0)
    F, Q_true = constant_velocity(dt=1.0, process_var=0.5, n_dims=1)
    H, R_true = position_measurement_model(order=1, n_dims=1, measurement_var=2.0)
    x0 = np.array([[0.0], [1.0]])

    z = _simulate(F, H, Q_true, R_true, x0, T=200, rng=rng)

    # Start from deliberately wrong noise guesses.
    result = em_kalman(
        z, F, H, Q=np.eye(2) * 0.01, R=np.eye(1) * 0.01, n_iter=40, tol=1e-8
    )

    ll = result.loglik_history
    assert len(ll) >= 2
    # Log-likelihood must be non-decreasing (EM guarantee), allowing tiny numerical slack.
    diffs = np.diff(ll)
    assert np.all(diffs >= -1e-6)
    # And it should improve overall.
    assert ll[-1] > ll[0]


def test_em_recovers_measurement_noise():
    rng = np.random.default_rng(1)
    F, Q_true = constant_velocity(dt=1.0, process_var=0.2, n_dims=1)
    H, _ = position_measurement_model(order=1, n_dims=1)
    R_true = np.array([[3.0]])
    x0 = np.array([[0.0], [0.5]])

    z = _simulate(F, H, Q_true, R_true, x0, T=500, rng=rng)

    result = em_kalman(
        z, F, H, Q=np.eye(2) * 0.05, R=np.eye(1) * 0.5, n_iter=60, tol=1e-8
    )

    # Learned R should land in the neighborhood of the true 3.0.
    assert np.isclose(result.R[0, 0], R_true[0, 0], rtol=0.35)


def test_em_learn_R_only_keeps_Q_fixed():
    rng = np.random.default_rng(2)
    F, _ = constant_velocity(dt=1.0, process_var=0.3, n_dims=1)
    H, _ = position_measurement_model(order=1, n_dims=1)
    z = _simulate(F, H, np.eye(2) * 0.3, np.array([[1.0]]), np.zeros((2, 1)), 100, rng)

    Q_fixed = np.eye(2) * 0.123
    result = em_kalman(z, F, H, Q=Q_fixed.copy(), R=np.eye(1), learn_Q=False, n_iter=10)
    assert np.allclose(result.Q, Q_fixed)


def test_em_converges_flag():
    rng = np.random.default_rng(3)
    F, Q = constant_velocity(dt=1.0, process_var=0.4, n_dims=1)
    H, R = position_measurement_model(order=1, n_dims=1, measurement_var=1.0)
    z = _simulate(F, H, Q, R, np.zeros((2, 1)), 150, rng)

    result = em_kalman(z, F, H, n_iter=100, tol=1e-2)
    # Should stop early on convergence rather than running all 100 iterations.
    assert result.converged
    assert result.n_iter_run < 100
