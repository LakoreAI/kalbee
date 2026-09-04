# Getting Started

## Installation

=== "pip"
    ```bash
    pip install kalbee
    ```

=== "uv (recommended)"
    ```bash
    uv pip install kalbee
    ```

=== "From source"
    ```bash
    git clone https://github.com/LakoreAI/kalbee.git
    cd kalbee
    pip install -e .
    ```

## Core Concepts

All filters in kalbee share a common interface through `BaseFilter`:

| Step | Method | What it does |
|---|---|---|
| 1. Init | `__init__()` | Set initial state $x$, covariance $P$, and model matrices |
| 2. Predict | `predict(dt)` | Propagate state forward: $\hat{x}_{k|k-1} = f(x_{k-1})$ |
| 3. Update | `update(z)` | Correct state with measurement: $\hat{x}_k = \hat{x}_{k|k-1} + K(z - h(\hat{x}))$ |

Additional base methods:

| Method | What it does |
|---|---|
| `predict_only(dt)` | Run predict without modifying filter state (useful for planning) |
| `reset(state, covariance)` | Reinitialize filter state without recreating the object |
| `filter_sequence(zs, dt, missing)` | Process a full measurement array with missing data handling |
| `save_state(path)` / `load_state(path)` | JSON serialization of filter state |

## Your First Filter

```python
import numpy as np
from kalbee import KalmanFilter

# State: [position, velocity], measuring position only
state = np.zeros((2, 1))
covariance = np.eye(2)

# Constant-velocity model (dt=1)
F = np.array([[1, 1], [0, 1]])  # Transition
Q = np.eye(2) * 0.01            # Process noise
H = np.array([[1, 0]])           # Measurement matrix
R = np.array([[0.1]])            # Measurement noise

kf = KalmanFilter(state, covariance, F, Q, H, R)

# Predict & update loop
measurements = [1.2, 2.1, 2.8, 4.1, 5.0]
for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))
    print(f"Position: {kf.x[0,0]:.2f}, Velocity: {kf.x[1,0]:.2f}")
```

## Quick Experiment

Compare multiple filters with one line:

```python
from kalbee import run_experiment

report = run_experiment(
    signal="sine",
    filters=["kf", "ekf", "ukf", "pf"],
    noise_std=0.5,
    duration=10.0,
)
print(report.summary())
```

## Using AutoFilter

Switch between filters without changing code structure:

```python
from kalbee import AutoFilter

kf = AutoFilter.from_filter(state, cov, F, Q, H, R, mode="kf")
ekf = AutoFilter.from_filter(state, cov, Q, R, mode="ekf")
ukf = AutoFilter.from_filter(state, cov, Q, R, f, h, mode="ukf")
```

Available modes: `kf`, `ekf`, `ukf`, `abg`, `pf`, `enkf`, `if`, `akf`, `hmf`, `srkf`, `imm`, `vkf`, `hinfinity`, `fading_memory`, `sigma_point_ukf`

## Batch Processing

Process a full measurement sequence with missing data support:

```python
import numpy as np
from kalbee import KalmanFilter

# measurements: (T, m) array, some values may be NaN
measurements = np.array([
    [1.0], [2.0], [np.nan], [4.0], [5.0]
])

state_history, cov_history = kf.filter_sequence(
    measurements, dt=1.0, missing=np.nan
)
```

When a measurement contains the `missing` value (or NaN), the filter runs **predict-only** for that step.

## State Persistence

Save and restore filter state:

```python
kf.save_state("filter_state.json")
# ... later ...
kf.load_state("filter_state.json")
```

## What's Next?

Explore each filter in detail:

- [Kalman Filter](filters/kalman_filter.md) — Start here for linear systems
- [Extended KF](filters/extended_kalman_filter.md) — Non-linear with Jacobians
- [Unscented KF](filters/unscented_kalman_filter.md) — Non-linear without Jacobians
- [H-Infinity Filter](filters/hinfinity_filter.md) — Robust worst-case estimation
- [Fading Memory KF](filters/fading_memory_kf.md) — Discounted covariance for tracking
- [SigmaPointUKF](filters/sigma_point_ukf.md) — Pluggable sigma point strategies
- [Particle Filter](filters/particle_filter.md) — Non-Gaussian distributions
- [Experiment Runner](features/experiments.md) — Compare all filters
