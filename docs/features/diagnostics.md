# Filter Diagnostics

Real-time monitoring and reporting for Kalman filter performance. Track NIS, NEES, innovation statistics, and covariance health during filter execution.

## FilterDiagnostics

Collects filter metrics at each time step and generates summary reports.

### Setup

```python
from kalbee import KalmanFilter, FilterDiagnostics

# Create filter
kf = KalmanFilter(state, cov, F, Q, H, R)

# Create diagnostics (m=measurement dim, n=state dim)
diag = FilterDiagnostics(m=1, n=2, alpha=0.05)
```

### Collecting Metrics

Call `collect()` after each predict-update cycle:

```python
for z in measurements:
    kf.predict()
    kf.update(z)

    # Collect diagnostics (optionally provide ground truth for NEES)
    snapshot = diag.collect(kf, measurement=z, ground_truth=true_state)

    print(f"Step {snapshot.timestamp}: NIS={snapshot.nis:.3f}, "
          f"Cov trace={snapshot.state_cov_trace:.4f}")
```

### Summary Report

```python
summary = diag.summary()
print(summary)
# {
#     'num_steps': 200,
#     'nis_mean': 1.05,
#     'nis_std': 1.42,
#     'nis_expected': 1.0,
#     'cov_trace_final': 0.234,
#     'cov_trace_mean': 0.567,
#     'nis_test_passed': True,
#     'nis_test_p_value': 0.342,
# }
```

### Consistency Check

```python
consistency = diag.check_consistency()
print(consistency)
# {
#     'nis_consistent': True,
#     'nis_mean': 1.05,
#     'nis_in_range': True,
# }
```

### Accessing History

```python
# Get all innovations as (T, m) array
innovations = diag.get_innovations()

# Get NIS values as array
nis_values = diag.get_nis_values()

# Reset diagnostics
diag.reset()
```

---

## FilterSnapshot

Each `collect()` call returns a `FilterSnapshot` dataclass:

```python
@dataclass
class FilterSnapshot:
    timestamp: int
    state_mean: np.ndarray       # Current state estimate
    state_cov_trace: float       # Trace of covariance (overall uncertainty)
    innovation: Optional[np.ndarray]
    innovation_cov: Optional[np.ndarray]
    nis: Optional[float]         # Normalized Innovation Squared
    nees: Optional[float]        # Normalized Estimation Error Squared
    kalman_gain_norm: Optional[float]
```

---

## Complete Example

```python
import numpy as np
from kalbee import KalmanFilter, FilterDiagnostics

# Setup
state = np.zeros((2, 1))
cov = np.eye(2) * 10.0
F = np.array([[1, 1], [0, 1]])
Q = np.eye(2) * 0.01
H = np.array([[1, 0]])
R = np.array([[0.5]])

kf = KalmanFilter(state, cov, F, Q, H, R)
diag = FilterDiagnostics(m=1, n=2)

# Generate data
np.random.seed(42)
T = 100
true_states = []
measurements = []

for k in range(T):
    true_state = np.array([[k * 0.1], [0.1]])
    true_states.append(true_state)
    measurements.append(
        H @ true_state + np.random.randn(1, 1) * np.sqrt(R[0, 0])
    )

# Run filter with diagnostics
for k, z in enumerate(measurements):
    kf.predict()
    kf.update(z)
    snapshot = diag.collect(kf, ground_truth=true_states[k])

# Analyze results
summary = diag.summary()
print(f"Steps: {summary['num_steps']}")
print(f"Mean NIS: {summary['nis_mean']:.3f} (expected: {summary['nis_expected']:.0f})")
print(f"Covariance trace (final): {summary['cov_trace_final']:.4f}")
print(f"Consistent: {summary.get('nis_test_passed', 'N/A')}")
```
