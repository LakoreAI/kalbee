"""
GPS + IMU loosely-coupled sensor fusion recipe.

The classic robotics/drone use case: a fast, noisy accelerometer drives
prediction every tick via the Kalman filter's control input, and a slow,
noisy GPS fix corrects position whenever it's available. This is a
"loosely-coupled" architecture — GPS position feeds a standard measurement
update, IMU acceleration feeds the control input `u` (no dedicated INS
mechanization needed).

Run: python examples/gps_imu_fusion.py
"""

import numpy as np

from kalbee import KalmanFilter
from kalbee.models import (
    constant_velocity,
    imu_velocity_control,
    position_measurement_model,
)


def main():
    rng = np.random.default_rng(0)

    n_dims = 2
    dt_imu = 0.02  # 50 Hz IMU
    gps_every = 25  # ~2 Hz GPS
    steps = 1000

    accel_noise_std = 0.05
    gps_noise_std = 1.5

    # --- Ground truth: a target that accelerates for a while, then coasts ---
    true_pos = np.zeros(n_dims)
    true_vel = np.array([1.0, 0.5])

    def true_accel_at(step):
        if step < 300:
            return np.array([0.4, -0.2])
        return np.zeros(n_dims)

    # --- Build the filter ---
    # F, Q: constant-velocity kinematics for the predict step.
    # B: maps a world-frame accelerometer reading onto [pos, vel] per axis.
    # H, R: GPS observes position directly.
    F, Q = constant_velocity(dt=dt_imu, process_var=0.02, n_dims=n_dims)
    B = imu_velocity_control(dt=dt_imu, n_dims=n_dims)
    H, R = position_measurement_model(
        order=1, n_dims=n_dims, measurement_var=gps_noise_std**2
    )

    x0 = np.zeros((2 * n_dims, 1))
    P0 = np.eye(2 * n_dims) * 100.0
    kf = KalmanFilter(x0, P0, F, Q, H, R, control_matrix=B)

    print(f"{'step':>5} {'true_pos':>18} {'fused_pos':>18} {'gps_fix':>10}")

    for step in range(steps):
        true_accel = true_accel_at(step)
        true_pos = true_pos + true_vel * dt_imu + 0.5 * true_accel * dt_imu**2
        true_vel = true_vel + true_accel * dt_imu

        # IMU tick: noisy accelerometer, already gravity-compensated/world-frame
        # (e.g. via an attitude filter — see quaternion_attitude_ekf.py).
        imu_reading = true_accel + rng.standard_normal(n_dims) * accel_noise_std
        kf.predict(u=imu_reading)

        gps_fired = step % gps_every == 0
        if gps_fired:
            gps_reading = (
                true_pos + rng.standard_normal(n_dims) * gps_noise_std
            ).reshape(-1, 1)
            kf.update(gps_reading)

        if step % 100 == 0:
            fused_pos = kf.state[[0, 2], 0]
            marker = "<-- GPS" if gps_fired else ""
            print(
                f"{step:>5} {np.array2string(true_pos, precision=2):>18} "
                f"{np.array2string(fused_pos, precision=2):>18} {marker:>10}"
            )

    final_error = np.linalg.norm(kf.state[[0, 2], 0] - true_pos)
    print(f"\nFinal position error: {final_error:.3f} (GPS noise std: {gps_noise_std})")


if __name__ == "__main__":
    main()
