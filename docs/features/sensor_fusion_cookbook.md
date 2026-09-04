# Sensor-Fusion Cookbook: GPS + IMU

The two most-searched-for real-world Kalman use cases — orientation from a
gyroscope/accelerometer, and position from GPS + an accelerometer — as
ready-made, tested building blocks instead of hand-derived matrices.

## 1. Quaternion attitude EKF (gyro + accelerometer)

Estimates 3D orientation as a unit quaternion `q = [w, x, y, z]` by fusing a
gyroscope (predict step) with an accelerometer's gravity-direction reading
(update step). This is the standard first stage of most IMU/AHRS pipelines,
and it's what turns a raw body-frame accelerometer reading into the
world-frame acceleration the GPS+IMU recipe below expects.

```python
import numpy as np
from kalbee import ExtendedKalmanFilter
from kalbee.models import (
    quaternion_normalize, attitude_transition, attitude_transition_jacobian,
    gravity_measurement, gravity_measurement_jacobian,
)

dt = 0.01
ekf = ExtendedKalmanFilter(
    state=np.array([[1.0], [0.0], [0.0], [0.0]]),  # identity orientation
    covariance=np.eye(4) * 0.1,
    transition_covariance=np.eye(4) * 1e-4,
    measurement_covariance=np.eye(3) * 0.05,  # see tuning note below
)

for gyro, accel in imu_stream:  # gyro: rad/s (3,); accel: unit-normalized (3, 1)
    ekf.predict(
        dt=dt,
        f=lambda x, dt: attitude_transition(x, dt, gyro),
        F=lambda x, dt: attitude_transition_jacobian(x, dt, gyro),
    )
    ekf.state = quaternion_normalize(ekf.state)  # renormalize every step

    ekf.update(accel, h=gravity_measurement, H=gravity_measurement_jacobian)
    ekf.state = quaternion_normalize(ekf.state)
```

Full runnable version: [`examples/quaternion_attitude_ekf.py`](https://github.com/LakoreAI/kalbee/blob/main/examples/quaternion_attitude_ekf.py).

**Why `attitude_transition_jacobian` is exact, not approximate.** Most EKF
process Jacobians are first-order linearizations. This one isn't: because
`q_{k+1} = q_k ⊗ Δq` (Hamilton product) is *linear* in `q_k` for a fixed,
gyro-derived delta quaternion `Δq`, the Jacobian is the exact closed-form
right-multiplication matrix of `Δq` — no approximation error from the process
model at all.

**Two things worth knowing before you tune this:**

1. **Yaw is unobservable from gravity alone.** Any two orientations that
   differ only by a rotation about the vertical axis produce the same
   predicted accelerometer reading, so gyro noise/bias on that axis
   random-walks uncorrected. Add a magnetometer (or GPS-course) update if you
   need yaw too.
2. **`R` is a tuning knob, not literally the accelerometer's noise spec.**
   This is a *naive* quaternion-state EKF (the raw 4-vector is the state), not
   an error-state/multiplicative EKF (MEKF), so its linearization only holds
   for small corrections. Plugging in the raw sensor noise variance can make
   updates overshoot the unit-quaternion manifold before renormalizing,
   degrading convergence. Inflating `R` a few times past spec (`0.05` for a
   `0.02` accelerometer noise std, in the example above) keeps updates small
   and the filter well-behaved — see `tests/test_attitude.py` for the
   before/after numbers.

## 2. GPS + IMU loosely-coupled fusion

A fast, noisy accelerometer drives the predict step every tick via the
filter's control input; a slow, noisy GPS fix corrects position via a
standard measurement update. No dedicated INS mechanization needed — this is
plain `KalmanFilter` plus one control-input matrix.

```python
import numpy as np
from kalbee import KalmanFilter
from kalbee.models import constant_velocity, imu_velocity_control, position_measurement_model

n_dims = 2
dt_imu = 0.02  # 50 Hz IMU

F, Q = constant_velocity(dt=dt_imu, process_var=0.02, n_dims=n_dims)
B = imu_velocity_control(dt=dt_imu, n_dims=n_dims)
H, R = position_measurement_model(order=1, n_dims=n_dims, measurement_var=1.5**2)

kf = KalmanFilter(np.zeros((4, 1)), np.eye(4) * 100.0, F, Q, H, R, control_matrix=B)

for tick, accel in enumerate(imu_stream):  # world-frame, gravity-compensated
    kf.predict(u=accel)
    if tick % 25 == 0:  # GPS fix arrives roughly every 25 IMU ticks
        kf.update(next(gps_stream))

x, y = kf.state[0, 0], kf.state[2, 0]
```

Full runnable version, including a dead-reckoning-vs-fused comparison:
[`examples/gps_imu_fusion.py`](https://github.com/LakoreAI/kalbee/blob/main/examples/gps_imu_fusion.py).

`imu_velocity_control(dt, n_dims)` builds the control matrix `B` that maps a
world-frame accelerometer reading directly onto a constant-velocity
`[position, velocity]` state (`B @ u` adds `0.5·dt²·a` to position and
`dt·a` to velocity, per axis) — pair it with the attitude EKF above to
rotate a raw body-frame accelerometer reading into the world frame and
subtract gravity before feeding it in as `u`.

## Deriving your own EKF Jacobians

Both recipes above ship exact or analytic Jacobians, but most systems don't
have one handy. `kalbee.numerical_jacobian` (and its
`numerical_transition_jacobian`/`numerical_measurement_jacobian` wrappers)
computes one via central finite differences, so you can go straight from a
transition/measurement *function* to a working EKF without deriving anything
by hand:

```python
from kalbee import numerical_transition_jacobian, numerical_measurement_jacobian

ekf.predict(
    dt=dt,
    f=my_transition_fn,
    F=lambda x, dt: numerical_transition_jacobian(my_transition_fn, x, dt),
)
ekf.update(z, h=my_measurement_fn, H=lambda x: numerical_measurement_jacobian(my_measurement_fn, x))
```

It's slower than an analytic Jacobian (one extra function evaluation per
state dimension) but removes the single biggest source of EKF setup bugs —
a hand-derived Jacobian with a sign or algebra error.
