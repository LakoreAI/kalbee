# Auto-Tuning

Automatically tune process noise ($Q$) and measurement noise ($R$) covariances from data using NIS-based optimization.

## Available Methods

### Iterative Tuning

Gradient-descent style optimization that adjusts $Q$ and $R$ until the mean NIS matches the expected value:

```python
import numpy as np
from kalbee import tune_kalman_filter

# measurements: (T, m) array
# F: transition matrix, H: measurement matrix
result = tune_kalman_filter(
    measurements=measurements,
    F=F, H=H,
    n_iter=50,
    learning_rate=0.01,
    target_nis_ratio=1.0,  # mean_NIS / m should be 1.0
    tol=1e-4,
)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Q:\n{result.Q}")
print(f"R:\n{result.R}")
print(f"NIS history: {result.nis_history}")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `Q_init` | `0.01 * I` | Initial process noise |
| `R_init` | `I` | Initial measurement noise |
| `n_iter` | `50` | Maximum iterations |
| `learning_rate` | `0.01` | Step size for updates |
| `target_nis_ratio` | `1.0` | Target: `mean_NIS / m` |
| `tol` | `1e-4` | Convergence tolerance |

### Quick Tuning

Single-pass tuning for fast results:

```python
from kalbee import quick_tune

Q, R = quick_tune(
    measurements=measurements,
    F=F, H=H,
    process_var=0.01,
    measurement_var=1.0,
)

print(f"Q:\n{Q}")
print(f"R:\n{R}")
```

---

## Complete Example

```python
import numpy as np
from kalbee import KalmanFilter, tune_kalman_filter, quick_tune

# Generate synthetic data
np.random.seed(42)
true_Q = np.eye(2) * 0.05
true_R = np.array([[0.8]])

F = np.array([[1, 1], [0, 1]])
H = np.array([[1, 0]])

# True state trajectory
T = 200
true_states = np.zeros((T, 2, 1))
for k in range(1, T):
    true_states[k] = F @ true_states[k-1] + np.random.multivariate_normal(
        np.zeros(2), true_Q
    ).reshape(2, 1)

# Noisy measurements
measurements = np.array([
    H @ s + np.random.multivariate_normal(np.zeros(1), true_R).reshape(1, 1)
    for s in true_states
]).squeeze(-1)  # (T, 1)

# Method 1: Iterative tuning
result = tune_kalman_filter(measurements, F, H, n_iter=50)
print(f"Iterative: Q diag = {np.diag(result.Q)}, R = {result.R[0,0]:.3f}")

# Method 2: Quick tuning
Q_quick, R_quick = quick_tune(measurements, F, H)
print(f"Quick:     Q diag = {np.diag(Q_quick)}, R = {R_quick[0,0]:.3f}")

# Compare with ground truth
print(f"True:      Q diag = {np.diag(true_Q)}, R = {true_R[0,0]:.3f}")
```

---

## When to Use

| ✅ Use auto-tuning when | ❌ Don't use when |
|---|---|
| Q/R are unknown | You have accurate noise models |
| You have enough measurement data | Very short data sequences |
| Filter performance is poor | Filter is already well-tuned |
| Rapid prototyping | Production systems with known physics |
