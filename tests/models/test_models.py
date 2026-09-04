import numpy as np

from kalbee.models import (
    constant_velocity,
    constant_acceleration,
    constant_turn,
    discrete_white_noise,
    position_measurement_model,
)


def test_constant_velocity_shapes_and_values():
    F, Q = constant_velocity(dt=0.5, process_var=2.0, n_dims=1)
    assert F.shape == (2, 2)
    assert np.allclose(F, [[1.0, 0.5], [0.0, 1.0]])
    assert np.allclose(Q, Q.T)  # symmetric


def test_constant_velocity_multidim_block_diagonal():
    F, Q = constant_velocity(dt=1.0, process_var=1.0, n_dims=3)
    assert F.shape == (6, 6)
    assert Q.shape == (6, 6)
    # Cross-axis coupling must be zero (block diagonal).
    assert F[0, 2] == 0.0 and F[2, 0] == 0.0


def test_constant_velocity_propagates_position():
    F, _ = constant_velocity(dt=2.0, n_dims=1)
    x = np.array([[0.0], [3.0]])  # position 0, velocity 3
    x_next = F @ x
    assert np.isclose(x_next[0, 0], 6.0)  # 0 + 3 * 2
    assert np.isclose(x_next[1, 0], 3.0)


def test_constant_acceleration_shapes():
    F, Q = constant_acceleration(dt=1.0, process_var=1.0, n_dims=2)
    assert F.shape == (6, 6)
    assert Q.shape == (6, 6)
    # position update includes 0.5 dt^2 acceleration term
    assert np.isclose(F[0, 2], 0.5)


def test_constant_turn_degenerates_to_cv():
    F_ct, _ = constant_turn(dt=1.0, turn_rate=0.0, process_var=1.0)
    F_cv, _ = constant_velocity(dt=1.0, process_var=1.0, n_dims=2)
    assert np.allclose(F_ct, F_cv)


def test_constant_turn_preserves_speed():
    # A coordinated turn should rotate the velocity vector but keep its norm.
    F, _ = constant_turn(dt=1.0, turn_rate=0.3)
    x = np.array([[0.0], [1.0], [0.0], [0.0]])  # [x, vx, y, vy], speed 1
    x_next = F @ x
    speed = np.hypot(x_next[1, 0], x_next[3, 0])
    assert np.isclose(speed, 1.0, atol=1e-9)


def test_discrete_white_noise_is_psd():
    Q = discrete_white_noise(order=1, dt=0.7, var=1.5)
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= -1e-12)


def test_position_measurement_model():
    H, R = position_measurement_model(order=1, n_dims=2, measurement_var=0.5)
    assert H.shape == (2, 4)
    assert R.shape == (2, 2)
    # H selects positions (indices 0 and 2) from [x, vx, y, vy]
    x = np.array([[10.0], [1.0], [20.0], [2.0]])
    assert np.allclose(H @ x, [[10.0], [20.0]])
    assert np.allclose(R, np.eye(2) * 0.5)
