import numpy as np

from kalbee.modules.utils.linalg import safe_inv, batched_inv, safe_cholesky


def test_safe_inv_well_conditioned():
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    inv = safe_inv(A)
    assert np.allclose(inv @ A, np.eye(2))


def test_safe_inv_singular_falls_back():
    # Exactly singular matrix: safe_inv must still return a finite result
    # via regularization / pseudo-inverse rather than raising.
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    inv = safe_inv(A)
    assert np.all(np.isfinite(inv))


def test_safe_inv_ill_conditioned_finite():
    # Near-singular: plain np.linalg.inv would return garbage without raising,
    # so the condition-number guard must route to the regularized path.
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-18]])
    inv = safe_inv(A)
    assert np.all(np.isfinite(inv))


def test_batched_inv_matches_per_matrix():
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((5, 3, 3))
    # Make each matrix well-conditioned (SPD)
    batch = batch @ np.swapaxes(batch, -1, -2) + np.eye(3)
    inv = batched_inv(batch)
    for i in range(5):
        assert np.allclose(inv[i], np.linalg.inv(batch[i]))


def test_batched_inv_singular_fallback():
    singular = np.tile(np.ones((2, 2)), (3, 1, 1))  # (3, 2, 2) all-ones
    inv = batched_inv(singular)
    assert inv.shape == (3, 2, 2)
    assert np.all(np.isfinite(inv))


def test_safe_cholesky_lower_and_upper():
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = safe_cholesky(A, lower=True)
    U = safe_cholesky(A, lower=False)
    assert np.allclose(L @ L.T, A)
    assert np.allclose(U.T @ U, A)
    assert np.allclose(U, L.T)


def test_safe_cholesky_psd_fallback():
    # Positive semi-definite (rank-deficient): raw cholesky fails, fallback adds reg.
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    L = safe_cholesky(A)
    assert np.all(np.isfinite(L))
