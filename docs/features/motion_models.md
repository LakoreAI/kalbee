# Motion & Measurement Models

Building a filter usually means hand-assembling the transition matrix $F$, the process-noise covariance $Q$, and the measurement pair $(H, R)$. The `kalbee.models` module ships ready-made builders for the most common kinematic models so you can skip the boilerplate.

All models use a **per-axis block convention**: for `n_dims` spatial axes and a kinematic `order` (1 = `[pos, vel]`, 2 = `[pos, vel, acc]`), the state is one contiguous kinematic block per axis:

$$
\mathbf{x} = [\,p_1, v_1, (a_1),\; p_2, v_2, (a_2),\; \dots\,]^\top
$$

## Constant Velocity (CV)

Per-axis state is $[position, velocity]$ with transition:

$$
F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}
$$

```python
from kalbee.models import constant_velocity

F, Q = constant_velocity(dt=1.0, process_var=0.1, n_dims=2)
# F, Q are 4x4 for a 2-D target: state = [x, vx, y, vy]
```

## Constant Acceleration (CA)

Per-axis state is $[position, velocity, acceleration]$:

$$
F = \begin{bmatrix} 1 & \Delta t & \tfrac{1}{2}\Delta t^2 \\ 0 & 1 & \Delta t \\ 0 & 0 & 1 \end{bmatrix}
$$

```python
from kalbee.models import constant_acceleration

F, Q = constant_acceleration(dt=1.0, process_var=0.1, n_dims=1)  # 3x3
```

## Constant Turn (CT)

A planar coordinated-turn model with a **known** turn rate $\omega$. State is ordered $[x, v_x, y, v_y]$. With a known turn rate the model is linear, so it works directly with the standard `KalmanFilter`:

$$
F = \begin{bmatrix}
1 & \frac{\sin\omega\Delta t}{\omega} & 0 & -\frac{1-\cos\omega\Delta t}{\omega} \\
0 & \cos\omega\Delta t & 0 & -\sin\omega\Delta t \\
0 & \frac{1-\cos\omega\Delta t}{\omega} & 1 & \frac{\sin\omega\Delta t}{\omega} \\
0 & \sin\omega\Delta t & 0 & \cos\omega\Delta t
\end{bmatrix}
$$

```python
from kalbee.models import constant_turn

F, Q = constant_turn(dt=1.0, turn_rate=0.3)  # 4x4
```

!!! tip "IMM pairing"
    Constant-velocity + constant-turn is the classic maneuvering-target pair for the [Interacting Multiple Model](../filters/interacting_multiple_model.md) estimator — one model for straight motion, one for turns.

As $\omega \to 0$ the model gracefully degenerates to constant velocity (avoiding a $0/0$).

## Measurement Model

`position_measurement_model` builds an $H$ that observes position on every axis, plus the matching $R$:

```python
from kalbee.models import position_measurement_model

H, R = position_measurement_model(order=1, n_dims=2, measurement_var=0.5)
# H is 2x4, selecting positions from [x, vx, y, vy]; R = 0.5 * I
```

## Process-Noise Helper

`discrete_white_noise` returns the single-axis discrete white-noise covariance used internally by the models — handy when composing custom models:

$$
Q_{\text{CV}} = \begin{bmatrix} \tfrac{\Delta t^4}{4} & \tfrac{\Delta t^3}{2} \\ \tfrac{\Delta t^3}{2} & \Delta t^2 \end{bmatrix} \sigma^2
$$

```python
from kalbee.models import discrete_white_noise

Q_axis = discrete_white_noise(order=1, dt=1.0, var=0.1)  # 2x2
```

## Putting It Together

```python
import numpy as np
from kalbee import KalmanFilter
from kalbee.models import constant_velocity, position_measurement_model

F, Q = constant_velocity(dt=1.0, process_var=0.1, n_dims=2)
H, R = position_measurement_model(order=1, n_dims=2, measurement_var=0.5)

x0 = np.zeros((4, 1))
kf = KalmanFilter(x0, np.eye(4) * 10.0, F, Q, H, R)
kf.predict()
kf.update(np.array([[1.2], [0.9]]))
```
