import numpy as np

from kalbee import AsyncSensorBuffer, KalmanFilter


def test_async_sensor_buffer_oosm():
    """AsyncSensorBuffer should process out-of-order measurements in chronological order."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])

    kf = KalmanFilter(np.zeros((2, 1)), np.eye(2) * 10.0, F, Q, H, R)
    buffer = AsyncSensorBuffer(kf, buffer_capacity=10)

    buffer.initialize(timestamp=0.0, state=np.zeros((2, 1)), covariance=np.eye(2))

    # Add measurements out of order: t=2.0 first, then t=1.0
    buffer.add_measurement(timestamp=2.0, measurement=np.array([[2.0]]))
    buffer.add_measurement(timestamp=1.0, measurement=np.array([[1.0]]))

    assert buffer.latest_state.shape == (2, 1)
