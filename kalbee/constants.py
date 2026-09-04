"""
Central registry of package-wide constants.

Keep values that are shared across modules (or that must stay in sync with the
package metadata) in one place so they only ever need to be changed once.
"""

import numpy as np

__all__ = [
    "__version__",
    "COND_LIMIT",
    "MATRIX_INV_REGULARIZATION",
    "BATCHED_INV_REGULARIZATION",
    "CHOLESKY_REGULARIZATION",
    "DEFAULT_INITIAL_COVARIANCE",
]

# ---------------------------------------------------------------------------
# Package version
#
# Single source of truth for ``kalbee.__version__`` and the ``--version`` CLI
# flag. Keep in sync with the ``[project] version`` in ``pyproject.toml``.
# ---------------------------------------------------------------------------
__version__ = "0.6.0"

# ---------------------------------------------------------------------------
# Linear-algebra numerical-stability policy (``kalbee.modules.utils.linalg``)
#
# Matrices with a condition number at or above ``COND_LIMIT`` are treated as
# effectively singular; direct inverses are only trusted below it. The
# ``*_REGULARIZATION`` values are the Tikhonov diagonals added as a fallback.
# ---------------------------------------------------------------------------
COND_LIMIT = 1.0 / float(np.finfo(float).eps)

# Diagonal regularization added to a (near-)singular matrix before retrying a
# direct inverse in ``safe_inv`` / ``batched_inv``.
MATRIX_INV_REGULARIZATION = 1e-6
BATCHED_INV_REGULARIZATION = 1e-6

# Diagonal regularization added to a positive semi-definite matrix before
# retrying the Cholesky factorization in ``safe_cholesky``.
CHOLESKY_REGULARIZATION = 1e-9

# ---------------------------------------------------------------------------
# State-estimation defaults
# ---------------------------------------------------------------------------

# Default diagonal magnitude of the initial covariance used when (re)initializing
# a filter without an explicit covariance (e.g. ``BaseFilter.reset`` and the
# ``KalmanEstimator`` / auto-tune entry points).
DEFAULT_INITIAL_COVARIANCE = 100.0
