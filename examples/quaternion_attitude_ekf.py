"""
Quaternion attitude (orientation) EKF recipe.

Fuses a gyroscope (angular velocity) with an accelerometer (gravity
direction) to estimate 3D orientation — the standard first stage of most
IMU/AHRS pipelines, and the piece that turns a raw body-frame accelerometer
reading into the world-frame acceleration ``gps_imu_fusion.py`` expects.

Note: gravity alone cannot observe rotation about the vertical (yaw) axis;
add a magnetometer update for full 3-axis observability.

Run: python examples/quaternion_attitude_ekf.py
"""

import numpy as np

from kalbee import ExtendedKalmanFilter
from kalbee.models import (
    quaternion_normalize,
    quaternion_angular_error,
    attitude_transition,
    attitude_transition_jacobian,
    gravity_measurement,
    gravity_measurement_jacobian,
)


def main():
    rng = np.random.default_rng(1)

    dt = 0.01
    steps = 500
    true_gyro = np.array(
        [0.05, 0.15, 0.0]
    )  # rad/s, roll+pitch only (yaw-observable-free case)
    gyro_noise_std = 0.01
    accel_noise_std = 0.02

    q_true = np.array([[1.0], [0.0], [0.0], [0.0]])  # identity orientation

    q0 = np.array([[1.0], [0.0], [0.0], [0.0]])
    P0 = np.eye(4) * 0.1
    Q = np.eye(4) * 1e-4
    # R is a tuning knob, not literally the sensor's noise variance: this is
    # a naive quaternion-state EKF, so an R this small lets single updates
    # overshoot the linear regime. Inflating it keeps updates well behaved
    # (see the module docstring in kalbee.models.attitude).
    R = np.eye(3) * 0.05

    ekf = ExtendedKalmanFilter(
        state=q0, covariance=P0, transition_covariance=Q, measurement_covariance=R
    )

    print(f"{'step':>5} {'true_quat':>32} {'est_quat':>32} {'err_deg':>8}")

    for step in range(steps):
        q_true = quaternion_normalize(attitude_transition(q_true, dt, true_gyro))

        measured_gyro = true_gyro + rng.standard_normal(3) * gyro_noise_std
        ekf.predict(
            dt=dt,
            f=lambda x, dt: attitude_transition(x, dt, measured_gyro),
            F=lambda x, dt: attitude_transition_jacobian(x, dt, measured_gyro),
        )
        ekf.state = quaternion_normalize(
            ekf.state
        )  # renormalize after every predict/update

        accel = (
            gravity_measurement(q_true) + rng.standard_normal((3, 1)) * accel_noise_std
        )
        ekf.update(accel, h=gravity_measurement, H=gravity_measurement_jacobian)
        ekf.state = quaternion_normalize(ekf.state)

        if step % 100 == 0:
            err_deg = np.degrees(quaternion_angular_error(ekf.state, q_true))
            print(
                f"{step:>5} {np.array2string(q_true.flatten(), precision=3):>32} "
                f"{np.array2string(ekf.state.flatten(), precision=3):>32} {err_deg:>8.2f}"
            )

    final_err_deg = np.degrees(quaternion_angular_error(ekf.state, q_true))
    print(f"\nFinal orientation error: {final_err_deg:.2f} degrees")


if __name__ == "__main__":
    main()
