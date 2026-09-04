# Fading Memory Kalman Filter

The **Fading Memory Kalman Filter** inflates the predicted covariance by a discounting factor $\alpha \geq 1$, preventing the filter from becoming overconfident in its motion model. This is especially useful for tracking maneuvering targets where the true dynamics may deviate from the assumed model.

## Fundamental Concepts

### The Problem

A standard Kalman Filter can "lock on" to a state estimate and ignore new measurements when the predicted covariance becomes too small. This happens when:

- The target maneuvers (changes velocity/acceleration)
- The model is slightly wrong
- The filter has been running for a long time without disturbances

### The Algorithm

**Predict** — standard KF predict with covariance inflation:

$$
\begin{aligned}
\hat{x}_{k|k-1} &= F \hat{x}_{k-1} \\
P_{k|k-1} &= \alpha (F P_{k-1} F^T) + Q
\end{aligned}
$$

**Update** — same as standard KF (Joseph form):

$$
\begin{aligned}
K_k &= P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1} \\
\hat{x}_k &= \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1}) \\
P_k &= (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T
\end{aligned}
$$

!!! note "Fading factor"
    - $\alpha = 1.0$: Standard Kalman Filter (no fading)
    - $\alpha = 1.01 - 1.1$: Typical range for tracking applications
    - $\alpha > 1.1$: Aggressive fading (use with caution)

### When to Use

| ✅ Use Fading Memory when | ❌ Don't use when |
|---|---|
| Target may maneuver | System model is perfectly known |
| Filter becomes overconfident | Small state dimension with short runs |
| Long tracking sequences | — |
| Adaptive behavior needed without full AKF | — |

---

## How to Use

### Basic Example

```python
import numpy as np
from kalbee import FadingMemoryKalmanFilter

state = np.zeros((2, 1))
covariance = np.eye(2) * 10.0

dt = 1.0
F = np.array([[1, dt], [0, 1]])
Q = np.eye(2) * 0.01
H = np.array([[1, 0]])
R = np.array([[0.5]])

# fading_factor=1.05: 5% covariance inflation per step
fading_kf = FadingMemoryKalmanFilter(
    state, covariance, F, Q, H, R, fading_factor=1.05
)

measurements = [1.2, 2.1, 2.8, 4.1, 5.0]
for z in measurements:
    fading_kf.predict(dt=dt)
    fading_kf.update(np.array([[z]]))
    print(f"Position: {fading_kf.x[0,0]:.2f}, Velocity: {fading_kf.x[1,0]:.2f}")
```

### Comparing Fading Factors

```python
from kalbee import KalmanFilter, FadingMemoryKalmanFilter

# No fading (standard KF)
kf_standard = KalmanFilter(state, cov, F, Q, H, R)

# Light fading
kf_light = FadingMemoryKalmanFilter(state, cov, F, Q, H, R, fading_factor=1.02)

# Heavy fading
kf_heavy = FadingMemoryKalmanFilter(state, cov, F, Q, H, R, fading_factor=1.10)
```

---

## Run an Experiment

```python
from kalbee import run_experiment

report = run_experiment(
    signal="sine",
    filters=["kf", "fading_memory"],
    noise_std=0.5,
    duration=10.0,
    seed=42,
)
print(report.summary())
```
