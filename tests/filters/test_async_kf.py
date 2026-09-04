import asyncio

import numpy as np

from kalbee import AsyncKalmanFilter, KalmanFilter


class TestAsyncKalmanFilter:
    """Tests for Async wrapper."""

    def test_basic_async(self):
        state = np.zeros((2, 1))
        cov = np.eye(2)
        F = np.array([[1, 1], [0, 1]])
        Q = np.eye(2) * 0.01
        H = np.array([[1, 0]])
        R = np.array([[0.5]])

        kf = KalmanFilter(state, cov, F, Q, H, R)
        async_kf = AsyncKalmanFilter(kf)

        async def run():
            await async_kf.predict(dt=1.0)
            return await async_kf.update(np.array([[1.0]]))

        result = asyncio.run(run())
        assert result.shape == (2, 1)
