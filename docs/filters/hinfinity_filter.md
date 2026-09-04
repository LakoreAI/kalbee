# H-Infinity Filter (H∞)

The **H-Infinity Filter** provides robust state estimation by minimizing the worst-case estimation error. Unlike the standard Kalman Filter which assumes known noise statistics, H-Infinity guarantees a bounded estimation error even under model uncertainty and unknown disturbance statistics.

## Fundamental Concepts

### The Problem

The H-Infinity filter solves a minimax problem: it minimizes the worst-case ratio of estimation error energy to disturbance energy:

$$\min_K \max_{w, v} \frac{\sum_{k} e_k^T e_k}{\sum_{k} (w_k^T w_k + v_k^T v_k)} < \gamma^2$$

Where $\gamma > 0$ is the **performance bound** — a user-specified guarantee on the worst-case L2-gain.

### The Algorithm

**Predict** — same as standard KF:

$$
\begin{aligned}
\hat{x}_{k|k-1} &= F \hat{x}_{k-1} \\
P_{k|k-1} &= F P_{k-1} F^T + Q
\end{aligned}
$$

**Update** — robust gain with $\gamma$-condition:

$$
\begin{aligned}
S &= H P_{k|k-1} H^T + R \\
K &= P_{k|k-1} (I - \gamma^{-2} P_{k|k-1})^{-1} H^T S^{-1} \\
\hat{x}_k &= \hat{x}_{k|k-1} + K (z_k - H \hat{x}_{k|k-1}) \\
P_k &= (I - K H) P_{k|k-1} (I - K H)^T + K R K^T
\end{aligned}
$$

!!! note "Gamma condition"
    The matrix $(I - \gamma^{-2} P)$ must be positive definite. If $\gamma$ is too small, the filter falls back to standard KF gain. As $\gamma \to \infty$, H-Infinity converges to the standard Kalman Filter.

### When to Use

| ✅ Use H-Infinity when | ❌ Don't use when |
|---|---|
| Noise statistics are uncertain | Perfect noise models are known (use KF) |
| Model uncertainties exist | System is well-modeled (use KF/EKF) |
| Worst-case guarantees needed | Average-case performance is sufficient |
| Adversarial disturbances present | Computational resources are very limited |

---

## How to Use

### Basic Example

```python
import numpy as np
from kalbee import HInfinityFilter

state = np.zeros((2, 1))
covariance = np.eye(2) * 10.0

dt = 1.0
F = np.array([[1, dt], [0, 1]])
Q = np.eye(2) * 0.01
H = np.array([[1, 0]])
R = np.array([[0.5]])

# gamma controls robustness: smaller = more robust, larger = closer to KF
hinfinity = HInfinityFilter(state, covariance, F, Q, H, R, gamma=5.0)

measurements = [1.2, 2.1, 2.8, 4.1, 5.0]
for z in measurements:
    hinfinity.predict(dt=dt)
    hinfinity.update(np.array([[z]]))
    print(f"Position: {hinfinity.x[0,0]:.2f}, Velocity: {hinfinity.x[1,0]:.2f}")
```

### Tuning Gamma

```python
# Conservative (more robust, slower convergence)
hinfinity_conservative = HInfinityFilter(state, cov, F, Q, H, R, gamma=2.0)

# Aggressive (less robust, faster convergence)
hinfinity_aggressive = HInfinityFilter(state, cov, F, Q, H, R, gamma=50.0)

# Equivalent to standard KF
hinfinity_kf = HInfinityFilter(state, cov, F, Q, H, R, gamma=1000.0)
```

### Control Input Support

Like the standard KF, H-Infinity supports control inputs via the $B$ matrix:

```python
B = np.array([[0.5], [1.0]])
hinfinity = HInfinityFilter(state, cov, F, Q, H, R, gamma=5.0, control_matrix=B)

u = np.array([[0.1]])  # Control input
hinfinity.predict(dt=dt, u=u)
```

---

## Run an Experiment

```python
from kalbee import run_experiment

report = run_experiment(
    signal="sine",
    filters=["kf", "hinfinity"],
    noise_std=0.5,
    duration=10.0,
    seed=42,
)
print(report.summary())
```
