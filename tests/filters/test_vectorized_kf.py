import numpy as np

from kalbee.modules.filters.vectorized_kf import VectorizedKalmanFilter


def test_vkf_initialization():
    batch_size = 5
    state = np.zeros((batch_size, 2, 1))
    covariance = np.repeat(np.eye(2)[np.newaxis, :, :], batch_size, axis=0)
    F = np.eye(2)
    Q = np.eye(2) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.eye(1) * 0.5

    # Expand matrices to batch shape for test
    F_batch = np.repeat(F[np.newaxis, :, :], batch_size, axis=0)
    Q_batch = np.repeat(Q[np.newaxis, :, :], batch_size, axis=0)
    H_batch = np.repeat(H[np.newaxis, :, :], batch_size, axis=0)
    R_batch = np.repeat(R[np.newaxis, :, :], batch_size, axis=0)

    vkf = VectorizedKalmanFilter(state, covariance, F_batch, Q_batch, H_batch, R_batch)

    assert np.array_equal(vkf.state, state)
    assert np.allclose(vkf.covariance, covariance)
    assert np.array_equal(vkf.x, state)
    assert np.allclose(vkf.P, covariance)


def test_vkf_predict():
    batch_size = 3
    dt = 1.0
    # Different initial states for different targets in the batch
    state = np.array([[[0.0], [1.0]], [[1.0], [2.0]], [[2.0], [3.0]]])
    covariance = np.repeat(np.eye(2)[np.newaxis, :, :], batch_size, axis=0)

    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.zeros((2, 2))  # No noise for predictable test
    H = np.array([[1.0, 0.0]])
    R = np.eye(1)

    F_batch = np.repeat(F[np.newaxis, :, :], batch_size, axis=0)
    Q_batch = np.repeat(Q[np.newaxis, :, :], batch_size, axis=0)
    H_batch = np.repeat(H[np.newaxis, :, :], batch_size, axis=0)
    R_batch = np.repeat(R[np.newaxis, :, :], batch_size, axis=0)

    vkf = VectorizedKalmanFilter(state, covariance, F_batch, Q_batch, H_batch, R_batch)
    vkf.predict()

    # Expected states:
    # Target 0: [0 + 1*1, 1] = [1, 1]
    # Target 1: [1 + 2*1, 2] = [3, 2]
    # Target 2: [2 + 3*1, 3] = [5, 3]
    expected_state = np.array([[[1.0], [1.0]], [[3.0], [2.0]], [[5.0], [3.0]]])
    assert np.allclose(vkf.state, expected_state)


def test_vkf_update():
    batch_size = 2
    state = np.array([[[0.0], [0.0]], [[0.0], [0.0]]])
    covariance = np.repeat((np.eye(2) * 10.0)[np.newaxis, :, :], batch_size, axis=0)

    F_batch = np.repeat(np.eye(2)[np.newaxis, :, :], batch_size, axis=0)
    Q_batch = np.repeat((np.eye(2) * 0.1)[np.newaxis, :, :], batch_size, axis=0)
    H_batch = np.repeat((np.array([[1.0, 0.0]]))[np.newaxis, :, :], batch_size, axis=0)
    R_batch = np.repeat((np.array([[0.1]]))[np.newaxis, :, :], batch_size, axis=0)

    vkf = VectorizedKalmanFilter(state, covariance, F_batch, Q_batch, H_batch, R_batch)

    # Different measurements for different batch targets
    measurements = np.array([[[5.0]], [[10.0]]])
    vkf.update(measurements)

    # Targets should move towards their respective measurements
    assert vkf.state[0, 0, 0] > 0
    assert vkf.state[0, 0, 0] < 5.0
    assert vkf.state[1, 0, 0] > 0
    assert vkf.state[1, 0, 0] < 10.0
