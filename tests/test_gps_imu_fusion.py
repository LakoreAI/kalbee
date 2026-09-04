import numpy as np

from kalbee import KalmanFilter
from kalbee.models import (
    constant_velocity,
    imu_velocity_control,
    position_measurement_model,
)
from kalbee.modules.utils.metrics import rmse


def test_imu_velocity_control_shape():
    B = imu_velocity_control(dt=0.1, n_dims=2)
    assert B.shape == (4, 2)


def test_imu_velocity_control_single_axis_matches_kinematics():
    dt = 0.1
    B = imu_velocity_control(dt=dt, n_dims=1)
    assert np.allclose(B, [[0.5 * dt**2], [dt]])


def test_gps_imu_loosely_coupled_fusion_tracks_accelerating_target():
    """
    A target accelerates in 2-D. A noisy "IMU" (fast, biased-free accel) drives
    the predict step every tick via the control input; a noisy "GPS" (slow)
    corrects position via the standard measurement update. The fused estimate
    should beat both raw GPS noise and pure IMU dead-reckoning.
    """
    rng = np.random.default_rng(0)
    n_dims = 2
    dt_imu = 0.02
    steps = 500
    gps_every = 25  # GPS update every 25 IMU ticks (0.5s)

    accel_noise_std = 0.05
    gps_noise_std = 1.0

    true_pos = np.zeros(n_dims)
    true_vel = np.array([1.0, 0.5])
    true_accel = np.array([0.3, -0.1])

    F, Q = constant_velocity(dt=dt_imu, process_var=0.01, n_dims=n_dims)
    B = imu_velocity_control(dt=dt_imu, n_dims=n_dims)
    H, R = position_measurement_model(
        order=1, n_dims=n_dims, measurement_var=gps_noise_std**2
    )

    x0 = np.zeros((2 * n_dims, 1))
    P0 = np.eye(2 * n_dims) * 100.0
    kf = KalmanFilter(x0, P0, F, Q, H, R, control_matrix=B)

    dead_reckon_pos = np.zeros(n_dims)
    dead_reckon_vel = np.zeros(n_dims)

    fused_errors = []
    dead_reckon_errors = []

    for step in range(steps):
        true_pos = true_pos + true_vel * dt_imu + 0.5 * true_accel * dt_imu**2
        true_vel = true_vel + true_accel * dt_imu

        imu_reading = true_accel + rng.standard_normal(n_dims) * accel_noise_std
        kf.predict(u=imu_reading)

        dead_reckon_pos = (
            dead_reckon_pos + dead_reckon_vel * dt_imu + 0.5 * imu_reading * dt_imu**2
        )
        dead_reckon_vel = dead_reckon_vel + imu_reading * dt_imu

        if step % gps_every == 0:
            gps_reading = (
                true_pos + rng.standard_normal(n_dims) * gps_noise_std
            ).reshape(-1, 1)
            kf.update(gps_reading)

        fused_pos = kf.state[[0, 2], 0]
        fused_errors.append(np.linalg.norm(fused_pos - true_pos))
        dead_reckon_errors.append(np.linalg.norm(dead_reckon_pos - true_pos))

    fused_rmse = rmse(np.array(fused_errors), np.zeros(steps))
    dead_reckon_rmse = rmse(np.array(dead_reckon_errors), np.zeros(steps))

    # Fused GPS+IMU should track far better than IMU-only dead reckoning,
    # which drifts unboundedly without any position correction.
    assert fused_rmse < dead_reckon_rmse
    assert fused_rmse < gps_noise_std
