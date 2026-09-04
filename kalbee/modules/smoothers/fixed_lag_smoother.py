from typing import Optional
import numpy as np
from collections import deque

from kalbee.modules.filters.base import BaseFilter


class FixedLagSmoother:
    """
    Fixed-Lag Smoother for real-time smoothed state estimation.

    Maintains a sliding window of the last N states and applies
    RTS smoothing within the window. Provides smoothed estimates
    with a fixed lag of L time steps.

    This is suitable for online applications where:
    - You need smoothed estimates in real-time
    - A small delay (L steps) is acceptable
    - You cannot wait for the full forward pass to complete

    Usage:
        smoother = FixedLagSmoother(filter_obj, lag=5)
        for z in measurements:
            smoother.predict(dt=1.0)
            smoothed_state = smoother.update(z)
    """

    def __init__(
        self,
        filter_obj: BaseFilter,
        lag: int = 5,
    ):
        """
        Initialize the Fixed-Lag Smoother.

        Args:
            filter_obj: BaseFilter instance (KF, EKF, UKF, etc.)
            lag: Number of steps to look back for smoothing (L).
        """
        self.filter = filter_obj
        self.lag = lag
        self.n = len(filter_obj.state)

        # Sliding window buffers
        self._filtered_states: deque = deque(maxlen=lag + 1)
        self._filtered_covs: deque = deque(maxlen=lag + 1)
        self._predicted_states: deque = deque(maxlen=lag + 1)
        self._predicted_covs: deque = deque(maxlen=lag + 1)

        # Smoothed output
        self._smoothed_state: Optional[np.ndarray] = None
        self._smoothed_cov: Optional[np.ndarray] = None

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step (delegates to underlying filter).

        Args:
            dt: Time step.

        Returns:
            Predicted state.
        """
        # Perform prediction first
        result = self.filter.predict(dt=dt, **kwargs)
        # Store predicted state and covariance (prior P_k|k-1)
        self._predicted_states.append(result.copy())
        self._predicted_covs.append(self.filter.P.copy())
        return result

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step with fixed-lag smoothing.

        Args:
            measurement: Measurement vector.

        Returns:
            Smoothed state estimate (lagged by L steps).
        """
        # Update the filter
        self.filter.update(measurement, **kwargs)

        # Store filtered state/cov
        self._filtered_states.append(self.filter.x.copy())
        self._filtered_covs.append(self.filter.P.copy())

        # If window is not full yet, return current filtered state
        if len(self._filtered_states) <= self.lag:
            self._smoothed_state = self.filter.x.copy()
            self._smoothed_cov = self.filter.P.copy()
            return self._smoothed_state

        # Run RTS smoothing on the window
        filtered = list(self._filtered_states)
        filtered_covs = list(self._filtered_covs)
        predicted = list(self._predicted_states)
        predicted_covs = list(self._predicted_covs)

        # Get transition matrix from filter if available
        F = getattr(self.filter, 'transition_matrix', None)

        if F is not None:
            # Use linear RTS smoother
            from kalbee.modules.smoothers.rts_smoother import RTSSmoother
            smoothed_states, smoothed_covs = RTSSmoother.smooth(
                filtered, filtered_covs, predicted, predicted_covs, F
            )
        else:
            # Use extended RTS smoother (numerical Jacobian)
            from kalbee.modules.smoothers.extended_rts_smoother import ExtendedRTSSmoother
            transition_fn = getattr(self.filter, 'transition_function', None)
            smoothed_states, smoothed_covs = ExtendedRTSSmoother.smooth(
                filtered, filtered_covs, predicted, predicted_covs,
                transition_function=transition_fn, state_dim=self.n
            )

        # Return the oldest smoothed state (the one that just "exits" the lag window)
        idx = max(0, len(smoothed_states) - self.lag - 1)
        self._smoothed_state = smoothed_states[idx]
        self._smoothed_cov = smoothed_covs[idx]

        return self._smoothed_state

    @property
    def smoothed_state(self) -> Optional[np.ndarray]:
        """Get the latest smoothed state."""
        return self._smoothed_state

    @property
    def smoothed_covariance(self) -> Optional[np.ndarray]:
        """Get the latest smoothed covariance."""
        return self._smoothed_cov

    @property
    def current_state(self) -> np.ndarray:
        """Get the current filtered (not smoothed) state."""
        return self.filter.x

    @property
    def current_covariance(self) -> np.ndarray:
        """Get the current filtered covariance."""
        return self.filter.P
