import numpy as np
from kalbee.modules.utils.metrics import rmse, nees, nis, log_likelihood


def test_rmse_perfect():
    """RMSE should be 0 for identical arrays."""
    estimated = np.array([[1.0], [2.0], [3.0]])
    truth = np.array([[1.0], [2.0], [3.0]])
    assert rmse(estimated, truth) == 0.0


def test_rmse_known_value():
    estimated = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 2.0, 4.0])
    # Error: [0, 0, -1], MSE = 1/3, RMSE = sqrt(1/3) ~ 0.577
    expected = np.sqrt(1.0 / 3.0)
    assert np.isclose(rmse(estimated, truth), expected)


def test_rmse_2d():
    estimated = np.array([[1.0, 0.0], [2.0, 0.0]])
    truth = np.array([[1.0, 1.0], [2.0, 1.0]])
    # Errors: [0, -1, 0, -1], MSE = 2/4 = 0.5, RMSE = sqrt(0.5)
    expected = np.sqrt(0.5)
    assert np.isclose(rmse(estimated, truth), expected)


def test_nees_consistent_filter():
    """For a well-tuned filter, NEES should be close to state dimension."""
    np.random.seed(42)
    n = 2
    T = 100

    state_errors = []
    covariances = []
    P = np.eye(n)

    for _ in range(T):
        e = np.random.multivariate_normal(np.zeros(n), P).reshape(-1, 1)
        state_errors.append(e)
        covariances.append(P.copy())

    nees_values = nees(state_errors, covariances)

    assert len(nees_values) == T
    # Average NEES should be approximately n=2
    avg_nees = np.mean(nees_values)
    assert np.isclose(avg_nees, n, atol=1.0)


def test_nis_basic():
    innovations = [np.array([[1.0]]), np.array([[2.0]])]
    S_matrices = [np.array([[1.0]]), np.array([[1.0]])]

    nis_values = nis(innovations, S_matrices)

    assert len(nis_values) == 2
    assert nis_values[0] == 1.0  # 1^T * 1^-1 * 1 = 1
    assert nis_values[1] == 4.0  # 2^T * 1^-1 * 2 = 4


def test_log_likelihood_basic():
    innovations = [np.array([[0.0]])]
    S_matrices = [np.array([[1.0]])]

    ll = log_likelihood(innovations, S_matrices)

    # -0.5 * (log(2*pi) + log(1) + 0) = -0.5 * log(2*pi)
    expected = -0.5 * np.log(2 * np.pi)
    assert np.isclose(ll, expected)


def test_log_likelihood_higher_for_better_model():
    """A model with smaller innovations should have higher log-likelihood."""
    S = [np.array([[1.0]])] * 5

    # Good model: small innovations
    good_innovations = [np.array([[0.1]])] * 5
    ll_good = log_likelihood(good_innovations, S)

    # Bad model: large innovations
    bad_innovations = [np.array([[5.0]])] * 5
    ll_bad = log_likelihood(bad_innovations, S)

    assert ll_good > ll_bad
