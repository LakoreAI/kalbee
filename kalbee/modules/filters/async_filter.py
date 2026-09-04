from typing import AsyncIterator, Callable, Optional
import numpy as np

from kalbee.modules.filters.base import BaseFilter


class AsyncKalmanFilter:
    """
    Async wrapper for Kalman filters for real-time streaming applications.

    Provides async predict/update methods for use with async sensors
    and event loops.

    Usage:
        async_filter = AsyncKalmanFilter(kf)

        async for measurement in sensor_stream:
            await async_filter.predict(dt=1.0)
            state = await async_filter.update(measurement)
    """

    def __init__(self, filter_obj: BaseFilter):
        """
        Wrap a filter object with async methods.

        Args:
            filter_obj: Any BaseFilter instance.
        """
        self._filter = filter_obj

    async def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """Async predict step."""
        return self._filter.predict(dt=dt, **kwargs)

    async def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """Async update step."""
        return self._filter.update(measurement, **kwargs)

    async def predict_update(
        self, measurement: np.ndarray, dt: float = 1.0, **kwargs
    ) -> np.ndarray:
        """Combined predict and update."""
        self._filter.predict(dt=dt, **kwargs)
        return self._filter.update(measurement, **kwargs)

    @property
    def state(self) -> np.ndarray:
        """Current state estimate."""
        return self._filter.x

    @property
    def covariance(self) -> np.ndarray:
        """Current covariance."""
        return self._filter.P

    @property
    def filter(self) -> BaseFilter:
        """Access underlying filter."""
        return self._filter


async def run_filter_stream(
    filter_obj: BaseFilter,
    measurements: AsyncIterator[np.ndarray],
    dt: float = 1.0,
    callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
) -> list:
    """
    Run a filter on an async measurement stream.

    Args:
        filter_obj: Any BaseFilter instance.
        measurements: Async iterator of measurements.
        dt: Time step.
        callback: Optional callback(state, covariance) after each update.

    Returns:
        List of state estimates.
    """
    async_filter = AsyncKalmanFilter(filter_obj)
    results = []

    async for z in measurements:
        await async_filter.predict(dt=dt)
        state = await async_filter.update(z)
        results.append(state.copy())

        if callback is not None:
            callback(state, filter_obj.P)

    return results
