"""
Sigma point generation strategies for the Unscented Kalman Filter.

Provides different methods for generating sigma points:
- SimplexSigmaPoints: Minimal set of 2n+1 points (default for UKF)
- MerweScaledSigmaPoints: Scalable points with tunable spread
- JulierSigmaPoints: Original Julier formulation

References:
    - Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation.
    - Merwe, R. V. D., & Wan, E. A. (2001). The unscented Kalman filter.
"""

import numpy as np

from kalbee.modules.utils.linalg import safe_cholesky


class SigmaPoints:
    """Base class for sigma point generation."""

    def __init__(self, n: int):
        self.n = n

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points.

        Args:
            x: State mean (n, 1) or (n,).
            P: State covariance (n, n).

        Returns:
            Sigma points array of shape (2n+1, n).
        """
        raise NotImplementedError

    @property
    def num_sigma_points(self) -> int:
        return 2 * self.n + 1

    @property
    def weights_mean(self) -> np.ndarray:
        """Weights for computing the weighted mean."""
        raise NotImplementedError

    @property
    def weights_cov(self) -> np.ndarray:
        """Weights for computing the weighted covariance."""
        raise NotImplementedError


class SimplexSigmaPoints(SigmaPoints):
    """
    Simplex sigma points - minimal 2n+1 points.

    This is the standard sigma point generation for UKF, producing
    2n+1 sigma points symmetrically distributed around the mean.
    """

    def __init__(self, n: int, alpha: float = 0.001, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize Simplex sigma points.

        Args:
            n: State dimension.
            alpha: Spread of sigma points (usually small, e.g. 1e-3).
            beta: Prior knowledge of distribution (2 is optimal for Gaussian).
            kappa: Secondary scaling parameter (usually 0).
        """
        super().__init__(n)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha ** 2 * (n + kappa) - n

        # Compute weights
        c = n + self.lambda_
        self._wm = np.full(2 * n + 1, 1.0 / (2 * c))
        self._wc = np.full(2 * n + 1, 1.0 / (2 * c))
        self._wm[0] = self.lambda_ / c
        self._wc[0] = self.lambda_ / c + (1 - alpha ** 2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points."""
        x = np.asarray(x, dtype=float).flatten()
        n = self.n

        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x

        # Cholesky factor
        c = n + self.lambda_
        S = safe_cholesky((c * P).astype(float), lower=False)

        for i in range(n):
            sigma_points[i + 1] = x + S[i]
            sigma_points[n + i + 1] = x - S[i]

        return sigma_points

    @property
    def weights_mean(self) -> np.ndarray:
        return self._wm

    @property
    def weights_cov(self) -> np.ndarray:
        return self._wc


class MerweScaledSigmaPoints(SigmaPoints):
    """
    Merwe scaled sigma points.

    A scalable sigma point formulation that provides better numerical
    properties for large state dimensions. The parameters alpha, beta,
    and kappa control the spread and weighting of sigma points.
    """

    def __init__(self, n: int, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize Merwe scaled sigma points.

        Args:
            n: State dimension.
            alpha: Controls spread of sigma points (0 < alpha <= 1).
            beta: Prior knowledge (2 is optimal for Gaussian).
            kappa: Secondary scaling (usually 0 or 3-n).
        """
        super().__init__(n)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute weights
        lam = alpha ** 2 * (n + kappa) - n
        c = n + lam

        self._wm = np.full(2 * n + 1, 1.0 / (2 * c))
        self._wc = np.full(2 * n + 1, 1.0 / (2 * c))
        self._wm[0] = lam / c
        self._wc[0] = lam / c + (1 - alpha ** 2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points."""
        x = np.asarray(x, dtype=float).flatten()
        n = self.n

        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x

        lam = self.alpha ** 2 * (n + self.kappa) - n
        S = safe_cholesky(((n + lam) * P).astype(float), lower=False)

        for i in range(n):
            sigma_points[i + 1] = x + S[i]
            sigma_points[n + i + 1] = x - S[i]

        return sigma_points

    @property
    def weights_mean(self) -> np.ndarray:
        return self._wm

    @property
    def weights_cov(self) -> np.ndarray:
        return self._wc


class JulierSigmaPoints(SigmaPoints):
    """
    Julier sigma points (original formulation).

    Uses a different weighting scheme that doesn't depend on the
    distribution parameters. More robust for certain applications.
    """

    def __init__(self, n: int, kappa: float = 0.0):
        """
        Initialize Julier sigma points.

        Args:
            n: State dimension.
            kappa: Scaling parameter (usually 0 or 3-n).
        """
        super().__init__(n)
        self.kappa = kappa

        # Julier weights
        self._wm = np.full(2 * n + 1, 1.0 / (2 * (n + kappa)))
        self._wc = np.full(2 * n + 1, 1.0 / (2 * (n + kappa)))
        self._wm[0] = kappa / (n + kappa)
        self._wc[0] = kappa / (n + kappa)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points."""
        x = np.asarray(x, dtype=float).flatten()
        n = self.n

        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x

        # Scale factor
        scale = n + self.kappa
        S = safe_cholesky((scale * P).astype(float), lower=False)

        for i in range(n):
            sigma_points[i + 1] = x + S[i]
            sigma_points[n + i + 1] = x - S[i]

        return sigma_points

    @property
    def weights_mean(self) -> np.ndarray:
        return self._wm

    @property
    def weights_cov(self) -> np.ndarray:
        return self._wc
