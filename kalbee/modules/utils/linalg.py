"""
Shared linear-algebra helpers with robust, condition-aware fallbacks.

These utilities centralize the numerical-stability policy that was previously
copy-pasted across every filter. The key subtlety they address: ``np.linalg.inv``
returns garbage (without raising) for *near*-singular matrices, so a plain
``try/except LinAlgError`` never triggers. We therefore guard on the condition
number before trusting a direct inverse.
"""

import numpy as np

# Machine epsilon threshold: matrices with condition number above this are
# treated as effectively singular.
_COND_LIMIT = 1.0 / np.finfo(float).eps


def safe_inv(A: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Robustly invert a square 2-D matrix.

    Strategy:
        1. If ``A`` is well-conditioned (finite condition number below the
           machine limit), return the direct inverse.
        2. Otherwise add ``reg * I`` and retry (Tikhonov regularization).
        3. As a last resort, return the Moore-Penrose pseudo-inverse.

    Args:
        A: Square matrix of shape (n, n).
        reg: Diagonal regularization added when ``A`` is ill-conditioned.

    Returns:
        An (n, n) inverse (or pseudo-inverse) of ``A``.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[-1]

    try:
        cond = np.linalg.cond(A)
        if np.isfinite(cond) and cond < _COND_LIMIT:
            return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        pass

    try:
        return np.linalg.inv(A + np.eye(n) * reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def batched_inv(A: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Invert a stack of square matrices of shape (batch, n, n).

    Unlike :func:`safe_inv`, this keeps the operation fully vectorized and only
    falls back to a single regularized retry if the batched inverse raises.

    Args:
        A: Stacked matrices of shape (batch, n, n).
        reg: Diagonal regularization added on the fallback path.

    Returns:
        Stacked inverses of shape (batch, n, n).
    """
    A = np.asarray(A, dtype=float)
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.inv(A + np.eye(A.shape[-1]) * reg)


def safe_cholesky(A: np.ndarray, reg: float = 1e-9, lower: bool = True) -> np.ndarray:
    """
    Cholesky factorization with a regularized fallback for matrices that are
    positive semi-definite (or slightly indefinite due to rounding).

    Args:
        A: Symmetric (near) positive-definite matrix of shape (n, n).
        reg: Diagonal regularization added when the factorization fails.
        lower: If ``True`` return the lower factor ``L`` (``A = L Lᵀ``);
               if ``False`` return the upper factor ``U`` (``A = Uᵀ U``).

    Returns:
        The requested Cholesky factor of shape (n, n).
    """
    A = np.asarray(A, dtype=float)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(A + np.eye(A.shape[-1]) * reg)
    return L if lower else L.T
