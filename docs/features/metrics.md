# Metrics & Diagnostics

kalbee provides standard filter performance metrics for evaluating estimation quality and filter consistency.

## Available Metrics

### RMSE — Root Mean Square Error

Measures how far estimates are from the truth:

$$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{k=1}^{T} (x_k - \hat{x}_k)^2}$$

```python
from kalbee import rmse
import numpy as np

estimated = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"RMSE: {rmse(estimated, truth):.4f}")
```

### NEES — Normalized Estimation Error Squared

Tests **filter consistency**. For a well-tuned filter, the average NEES should be approximately equal to the state dimension $n$:

$$\text{NEES}_k = (x_k - \hat{x}_k)^T P_k^{-1} (x_k - \hat{x}_k)$$

```python
from kalbee import nees
import numpy as np

# Simulate consistent errors
state_errors = [np.array([[0.1], [0.05]]) for _ in range(100)]
covariances = [np.eye(2) * 0.1 for _ in range(100)]

nees_values = nees(state_errors, covariances)
print(f"Average NEES: {np.mean(nees_values):.2f}  (should be ≈ 2)")
```

### NIS — Normalized Innovation Squared

Tests **measurement consistency**. Average NIS should be ≈ measurement dimension $m$:

$$\text{NIS}_k = v_k^T S_k^{-1} v_k$$

```python
from kalbee import nis

innovations = [np.array([[0.5]]), np.array([[0.3]]), np.array([[-0.2]])]
S_matrices = [np.array([[1.0]])] * 3

nis_values = nis(innovations, S_matrices)
print(f"NIS values: {nis_values}")
```

### Log-Likelihood

For **model comparison** — higher is better:

$$\mathcal{L} = -\frac{1}{2} \sum_k \left[ m \log(2\pi) + \log|S_k| + v_k^T S_k^{-1} v_k \right]$$

```python
from kalbee import log_likelihood

ll = log_likelihood(innovations, S_matrices)
print(f"Log-likelihood: {ll:.4f}")
```

## Formal Consistency Tests

For rigorous statistical testing, use the consistency test functions:

```python
from kalbee import nis_test, nees_test, innovation_whiteness_test

# NIS test (tests if innovations follow chi-squared(m))
nis_passed, nis_values, mean_nis, expected, p_value = nis_test(
    innovations, innovation_covariances, alpha=0.05
)

# NEES test (tests if state errors follow chi-squared(n))
nees_passed, nees_values, mean_nees, expected, p_value = nees_test(
    state_errors, covariances, alpha=0.05
)

# Whiteness test (tests if innovations are uncorrelated)
white_passed, autocorrelations = innovation_whiteness_test(
    innovations, max_lag=10, alpha=0.05
)
```

## Real-Time Diagnostics

For continuous monitoring during filter execution:

```python
from kalbee import FilterDiagnostics

diag = FilterDiagnostics(m=1, n=2)

for z in measurements:
    kf.predict()
    kf.update(z)
    snapshot = diag.collect(kf, ground_truth=true_state)

# Generate summary
summary = diag.summary()
```

See [Diagnostics](diagnostics.md) for full details.

## Using with Experiments

```python
from kalbee import run_experiment

report = run_experiment(
    signal="sine",
    filters=["kf", "ekf", "ukf"],
    noise_std=0.5,
    seed=42,
)

# Each result has built-in metrics
for result in report.results:
    print(f"{result.filter_name}: "
          f"RMSE={result.position_rmse():.4f}  "
          f"NEES={result.average_nees():.2f}")
```
