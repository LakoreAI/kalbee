# Consistency Tests

Formal hypothesis tests to verify whether a Kalman filter is correctly tuned. A consistent filter should pass these tests on well-generated data.

## Available Tests

### NIS Test — Normalized Innovation Squared

Tests whether innovations follow a chi-squared distribution under the null hypothesis that the filter is correctly tuned.

$$H_0: \text{NIS}_k \sim \chi^2(m)$$

Where $m$ is the measurement dimension. The expected mean NIS is $m$.

```python
import numpy as np
from kalbee import nis_test

# innovations: list of (m, 1) arrays
# innovation_covariances: list of (m, m) arrays
passed, nis_values, mean_nis, expected_mean, p_value = nis_test(
    innovations, innovation_covariances, alpha=0.05
)

print(f"Mean NIS: {mean_nis:.3f} (expected: {expected_mean:.1f})")
print(f"P-value: {p_value:.4f}")
print(f"Consistent: {passed}")
```

**Interpretation:**

- `passed = True`: Filter is consistent (innovations match expected statistics)
- `passed = False`: Filter may be overconfident or underconfident
- `mean_nis ≈ m`: Well-tuned
- `mean_nis >> m`: Underconfident (Q or R too small)
- `mean_nis << m`: Overconfident (Q or R too large)

### NEES Test — Normalized Estimation Error Squared

Tests whether state estimation errors follow a chi-squared distribution. Requires ground truth.

$$H_0: \text{NEES}_k \sim \chi^2(n)$$

Where $n$ is the state dimension.

```python
from kalbee import nees_test

# state_errors: list of (n, 1) arrays (truth - estimate)
# covariances: list of (n, n) arrays
passed, nees_values, mean_nees, expected_mean, p_value = nees_test(
    state_errors, covariances, alpha=0.05
)

print(f"Mean NEES: {mean_nees:.3f} (expected: {expected_mean:.1f})")
print(f"Consistent: {passed}")
```

### Innovation Whiteness Test

Tests whether innovations are white (uncorrelated over time). If the filter is optimal, innovations should be white noise.

$$H_0: \rho_k(\tau) = 0 \quad \text{for all lags } \tau \neq 0$$

```python
from kalbee import innovation_whiteness_test

passed, autocorrelations = innovation_whiteness_test(
    innovations, max_lag=10, alpha=0.05
)

print(f"Autocorrelations: {autocorrelations}")
print(f"White innovations: {passed}")
```

**Interpretation:**

- `passed = True`: No significant autocorrelation → filter is optimal
- `passed = False`: Correlations detected → filter model may be wrong

---

## Complete Example

```python
import numpy as np
from kalbee import KalmanFilter, nis_test, nees_test, innovation_whiteness_test

# Setup
state = np.zeros((2, 1))
cov = np.eye(2) * 10.0
F = np.array([[1, 1], [0, 1]])
Q = np.eye(2) * 0.01
H = np.array([[1, 0]])
R = np.array([[0.5]])

kf = KalmanFilter(state, cov, F, Q, H, R)

# Generate truth and measurements
np.random.seed(42)
true_states = []
measurements = []
for k in range(200):
    true_state = np.array([[k * 0.1], [0.1]])
    true_states.append(true_state)
    measurements.append(H @ true_state + np.random.randn(1, 1) * np.sqrt(R[0,0]))

# Run filter and collect innovations
innovations = []
innovation_covs = []
state_errors = []
covariances = []

for z in measurements:
    kf.predict()
    kf.update(z)

    innovations.append(kf.last_y)
    innovation_covs.append(kf.last_S)
    state_errors.append(true_states[len(innovations)-1] - kf.x)
    covariances.append(kf.P)

# Run consistency tests
nis_passed, _, nis_mean, nis_expected, _ = nis_test(innovations, innovation_covs)
nees_passed, _, nees_mean, nees_expected, _ = nees_test(state_errors, covariances)
white_passed, _ = innovation_whiteness_test(innovations)

print(f"NIS Test:  mean={nis_mean:.2f} (expect {nis_expected:.0f}), passed={nis_passed}")
print(f"NEES Test: mean={nees_mean:.2f} (expect {nees_expected:.0f}), passed={nees_passed}")
print(f"Whiteness: passed={white_passed}")
```
