# RTS Smoother

The **Rauch-Tung-Striebel (RTS) Smoother** is a post-processing algorithm that runs backward over the forward-pass results of a Kalman Filter to produce **optimally smoothed** state estimates.

## Fundamental Concepts

### Forward Filtering vs. Smoothing

| | Filtering | Smoothing |
|---|---|---|
| Uses | Past + current data | Past + current + **future** data |
| Processing | Real-time (online) | Post-processing (offline) |
| Accuracy | Good | **Better** (always ≤ filtering error) |

The RTS smoother takes the forward KF results and runs a **backward pass** to incorporate future information into every estimate:

$$
\begin{aligned}
G_k &= P_{k|k} F^T P_{k+1|k}^{-1} \\
\hat{x}_{k|N} &= \hat{x}_{k|k} + G_k (\hat{x}_{k+1|N} - \hat{x}_{k+1|k}) \\
P_{k|N} &= P_{k|k} + G_k (P_{k+1|N} - P_{k+1|k}) G_k^T
\end{aligned}
$$

### When to Use

| ✅ Use RTS when | ❌ Don't use when |
|---|---|
| Processing recorded data | Need real-time estimates |
| Want best possible accuracy | Data is streaming (use filter only) |
| Post-flight analysis | Memory is extremely limited |

## How to Use

```python
import numpy as np
from kalbee import KalmanFilter, RTSSmoother

# Setup
dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
Q = np.eye(2) * 0.01
H = np.array([[1.0, 0.0]])
R = np.array([[1.0]])

state = np.array([[0.0], [1.0]])
cov = np.eye(2) * 10.0
kf = KalmanFilter(state, cov, F, Q, H, R)

# Step 1: Forward pass — store everything
filtered_states, filtered_covs = [], []
predicted_states, predicted_covs = [], []

np.random.seed(42)
for t in range(20):
    kf.predict(dt=dt)
    predicted_states.append(kf.state.copy())
    predicted_covs.append(kf.covariance.copy())

    z = np.array([[float(t + 1) + np.random.randn()]])
    kf.update(z)
    filtered_states.append(kf.state.copy())
    filtered_covs.append(kf.covariance.copy())

# Step 2: Backward pass — smooth
smoothed_states, smoothed_covs = RTSSmoother.smooth(
    filtered_states, filtered_covs,
    predicted_states, predicted_covs,
    F,
)

# Compare
for k in [0, 5, 10, 15, 19]:
    f_err = abs(filtered_states[k][0, 0] - (k + 1))
    s_err = abs(smoothed_states[k][0, 0] - (k + 1))
    print(f"t={k+1:2d}  Filtered error: {f_err:.3f}  "
          f"Smoothed error: {s_err:.3f}  "
          f"Improvement: {(f_err - s_err) / f_err * 100:.0f}%")
```

!!! tip "Smoothing always helps"
    The smoothed covariance is always ≤ the filtered covariance (element-wise). The improvement is most dramatic at the **beginning** of the sequence, where the filter had less data.
