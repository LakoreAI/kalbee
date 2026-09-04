# Outlier Detection

Real-time detection of measurement outliers using chi-squared gating on the Normalized Innovation Squared (NIS).

## Chi2OutlierDetector

Detects outliers by comparing NIS against a chi-squared threshold. Supports both fixed and adaptive thresholds.

### Setup

```python
from kalbee import Chi2OutlierDetector

# Fixed threshold
detector = Chi2OutlierDetector(m=1, confidence=0.95)

# Adaptive threshold (adjusts based on recent NIS history)
adaptive_detector = Chi2OutlierDetector(
    m=1, confidence=0.95,
    adaptive=True,
    window_size=50,
    scale_factor=3.0,
)
```

### Checking Measurements

```python
import numpy as np

innovation = np.array([[2.5]])
innovation_cov = np.array([[1.0]])

result = detector.check(innovation, innovation_cov)

print(f"Is inlier: {result.is_inlier}")
print(f"NIS: {result.nis_value:.3f}")
print(f"Threshold: {result.threshold:.3f}")
print(f"Confidence: {result.confidence}")
```

### Batch Processing

```python
innovations = np.array([[0.5], [1.2], [8.0], [0.3], [0.8]])  # (T, m)
innovation_covs = np.array([np.eye(1)] * 5)                    # (T, m, m)

results = detector.batch_check(innovations, innovation_covs)

for i, r in enumerate(results):
    status = "✓" if r.is_inlier else "✗ OUTLIER"
    print(f"Step {i}: NIS={r.nis_value:.3f} {status}")
```

### Statistics

```python
stats = detector.get_statistics()
print(f"Checks: {stats['num_checks']}")
print(f"NIS mean: {stats['nis_mean']:.3f}")
print(f"Current threshold: {stats['current_threshold']:.3f}")
```

---

## DetectionResult

Each `check()` call returns a `DetectionResult` dataclass:

```python
@dataclass
class DetectionResult:
    is_inlier: bool
    nis_value: float
    threshold: float
    confidence: float
    innovation: Optional[np.ndarray]
```

---

## Adaptive Threshold

When `adaptive=True`, the threshold is computed as:

$$\text{threshold} = \mu_{\text{NIS}} + s \cdot \sigma_{\text{NIS}}$$

Where:

- $\mu_{\text{NIS}}$: Mean of recent NIS values (sliding window)
- $\sigma_{\text{NIS}}$: Standard deviation of recent NIS values
- $s$: `scale_factor` (default 3.0)

The adaptive threshold activates after `window_size` (default 50) measurements have been collected.

---

## Complete Example

```python
import numpy as np
from kalbee import KalmanFilter, Chi2OutlierDetector

# Setup
state = np.zeros((2, 1))
cov = np.eye(2) * 10.0
F = np.array([[1, 1], [0, 1]])
Q = np.eye(2) * 0.01
H = np.array([[1, 0]])
R = np.array([[0.5]])

kf = KalmanFilter(state, cov, F, Q, H, R)
detector = Chi2OutlierDetector(m=1, confidence=0.95, adaptive=True)

# Simulate with outliers
np.random.seed(42)
true_positions = np.arange(0, 10, 0.1)
outlier_indices = {15, 45, 75}  # Inject outliers

inliers = 0
outliers = 0

for k, pos in enumerate(true_positions):
    kf.predict()

    # Generate measurement (with occasional outlier)
    if k in outlier_indices:
        z = np.array([[pos + np.random.randn() * 10]])  # Large outlier
    else:
        z = np.array([[pos + np.random.randn() * 0.5]])  # Normal noise

    # Check for outlier
    innovation = z - kf.measure()
    S = H @ kf.P @ H.T + R
    result = detector.check(innovation, S)

    if result.is_inlier:
        kf.update(z)
        inliers += 1
    else:
        print(f"Step {k}: Outlier detected (NIS={result.nis_value:.2f})")
        outliers += 1

print(f"Inliers: {inliers}, Outliers: {outliers}")
```
