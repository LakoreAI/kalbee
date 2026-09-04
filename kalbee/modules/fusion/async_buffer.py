from typing import List, Tuple, Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter


class AsyncSensorBuffer:
    """
    Asynchronous Sensor Buffer for Out-Of-Sequence Measurements (OOSM).

    Handles out-of-order or variable-rate sensor inputs (e.g. GPS, IMU, LiDAR)
    by maintaining a history window of states and replaying measurements
    in chronological order when delayed data arrives.
    """

    def __init__(
        self,
        filter_obj: BaseFilter,
        buffer_capacity: int = 50,
        dt_default: float = 0.1,
    ):
        """
        Args:
            filter_obj: Instance of a filter subclass (e.g. KalmanFilter).
            buffer_capacity: Maximum number of historical measurements to store.
            dt_default: Default time delta if timestamp step is not specified.
        """
        self.filter = filter_obj
        self.buffer_capacity = buffer_capacity
        self.dt_default = dt_default

        # Timestamped measurements: list of (timestamp, measurement_vector, kwargs_dict)
        self._history: List[Tuple[float, np.ndarray, dict]] = []
        # Initial state checkpoint: (timestamp, state_vector, covariance_matrix)
        self._initial_checkpoint: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

    def initialize(self, timestamp: float, state: np.ndarray, covariance: np.ndarray):
        """Set initial timestamped filter state."""
        self._initial_checkpoint = (
            timestamp,
            state.copy(),
            covariance.copy(),
        )
        self.filter.state = state.copy()
        self.filter.covariance = covariance.copy()
        self._history.clear()

    def add_measurement(
        self, timestamp: float, measurement: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add a measurement with its timestamp (can be out-of-order).

        Replays prediction and update cycles in chronological order to bring
        the filter up to the latest timestamp.

        Args:
            timestamp: Time of observation.
            measurement: Measurement vector.
            **kwargs: Extra filter kwargs (e.g. u, B, F).

        Returns:
            Tuple of (latest_state, latest_covariance).
        """
        z = np.asanyarray(measurement, dtype=float)
        self._history.append((timestamp, z, kwargs))
        self._history.sort(key=lambda item: item[0])

        # Trim buffer if exceeding capacity
        if len(self._history) > self.buffer_capacity:
            oldest_ts, _, _ = self._history.pop(0)

        # Replay from checkpoint or oldest stored measurement
        self._replay()

        return self.filter.state, self.filter.covariance

    def _replay(self):
        """Replay all stored measurements from the initial checkpoint."""
        if not self._history:
            return

        if self._initial_checkpoint is not None:
            init_ts, init_state, init_cov = self._initial_checkpoint
            self.filter.state = init_state.copy()
            self.filter.covariance = init_cov.copy()
            current_time = init_ts
        else:
            current_time = self._history[0][0]

        for ts, z, kwargs in self._history:
            dt = max(1e-6, ts - current_time) if current_time < ts else self.dt_default
            self.filter.predict(dt=dt, **kwargs)
            self.filter.update(z, **kwargs)
            current_time = ts

    @property
    def latest_state(self) -> np.ndarray:
        return self.filter.state

    @property
    def latest_covariance(self) -> np.ndarray:
        return self.filter.covariance
