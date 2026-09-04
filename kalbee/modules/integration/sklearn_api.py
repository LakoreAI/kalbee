"""
scikit-learn-compatible wrapper around kalbee's Kalman filters.

Lets you drop a filter into an sklearn ``Pipeline``, grid-search its
hyperparameters, and gives users used to the ``fit``/``transform`` convention
a one-liner smoothing API::

    from kalbee.modules.integration.sklearn_api import KalmanEstimator
    smoothed = KalmanEstimator(n_dims=1).fit_transform(noisy_series)

Requires scikit-learn: ``pip install kalbee[sklearn]``.
"""

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted
except ImportError as exc:  # pragma: no cover - exercised only without sklearn
    raise ImportError(
        "scikit-learn is required for KalmanEstimator. "
        "Install it with: pip install kalbee[sklearn]"
    ) from exc

from kalbee.modules.filters.auto_filter import AutoFilter
from kalbee.models.motion import constant_velocity, constant_acceleration
from kalbee.models.measurement import position_measurement_model
from kalbee.modules.learning.auto_tune import quick_tune


class KalmanEstimator(BaseEstimator, TransformerMixin):
    """
    scikit-learn-style Kalman filter/smoother for measurement sequences.

    Wraps any kalbee filter behind ``fit``/``transform``/``predict`` so it can
    sit inside an ``sklearn.pipeline.Pipeline`` or be swept with
    ``GridSearchCV``. Each row of ``X`` is one time step; each column is one
    measured spatial axis (e.g. columns ``[x, y]`` for 2-D position).

    Args:
        mode: Filter to build — any :class:`~kalbee.AutoFilter` mode
            (``"kf"``, ``"akf"``, ``"srkf"``, ...).
        order: Kinematic order of the underlying motion model
            (1 = constant-velocity, 2 = constant-acceleration).
        dt: Time step between rows of ``X``.
        process_var: Process-noise variance for the default motion model.
        measurement_var: Measurement-noise variance for the default
            measurement model.
        tune: If True, auto-tune ``Q``/``R`` from ``X`` via
            :func:`~kalbee.quick_tune` instead of using ``process_var``/
            ``measurement_var`` directly. Only supported for ``mode="kf"``.
        return_full_state: If True, ``transform`` returns the full state
            vector at each step instead of just the position components.

    Example:
        >>> import numpy as np
        >>> from kalbee.modules.integration.sklearn_api import KalmanEstimator
        >>> t = np.linspace(0, 10, 100)
        >>> noisy = np.sin(t) + np.random.normal(0, 0.2, 100)
        >>> smoothed = KalmanEstimator(dt=t[1] - t[0]).fit_transform(noisy)
    """

    def __init__(
        self,
        mode: str = "kf",
        order: int = 1,
        dt: float = 1.0,
        process_var: float = 1.0,
        measurement_var: float = 1.0,
        tune: bool = False,
        return_full_state: bool = False,
    ):
        self.mode = mode
        self.order = order
        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.tune = tune
        self.return_full_state = return_full_state

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def fit(self, X, y=None):
        """Build the underlying filter for the shape of ``X``. Does not fit any state."""
        X = self._as_2d(X)
        n_dims = X.shape[1]

        model_builder = constant_acceleration if self.order == 2 else constant_velocity
        F, Q = model_builder(dt=self.dt, process_var=self.process_var, n_dims=n_dims)
        H, R = position_measurement_model(
            order=self.order, n_dims=n_dims, measurement_var=self.measurement_var
        )

        if self.tune:
            Q, R = quick_tune(X, F, H)

        n_state = F.shape[0]
        self._x0_ = np.zeros((n_state, 1))
        self._P0_ = np.eye(n_state) * 100.0

        self.filter_ = AutoFilter.from_filter(
            self._x0_, self._P0_, F, Q, H, R, mode=self.mode
        )
        self.n_features_in_ = n_dims
        self._position_idx_ = [i * (self.order + 1) for i in range(n_dims)]
        return self

    def transform(self, X):
        """Run predict/update over every row of ``X`` and return the filtered states.

        Re-runs from the initial state each call, so repeated calls on the
        same or different ``X`` are independent (no carry-over state).
        """
        check_is_fitted(self, "filter_")
        X = self._as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )

        self.filter_.reset(self._x0_, self._P0_)
        state_history, _ = self.filter_.filter_sequence(X, dt=self.dt)

        if self.return_full_state:
            return state_history[:, :, 0]
        return state_history[:, self._position_idx_, 0]

    def predict(self, X):
        """Alias for :meth:`transform`, for sklearn ``Pipeline``/estimator compatibility."""
        return self.transform(X)
