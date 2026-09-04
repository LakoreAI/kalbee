# Innovation Gating

Innovation gating rejects measurements that are statistically inconsistent with the filter's predictions. This prevents outliers from corrupting the state estimate.

## Available Methods

### Chi-Squared Gate

Tests whether the Normalized Innovation Squared (NIS) exceeds a chi-squared threshold:

$$\text{NIS} = v_k^T S_k^{-1} v_k \leq \chi^2_{m, \alpha}$$

Where $m$ is the measurement dimension and $\alpha$ is the confidence level.

```python
import numpy as np
from kalbee import chi2_gate

innovation = np.array([[0.5]])
innovation_cov = np.array([[1.0]])

passed, nis_value, threshold = chi2_gate(
    innovation, innovation_cov, confidence=0.95
)
print(f"NIS: {nis_value:.3f}, Threshold: {threshold:.3f}, Passed: {passed}")
```

### Mahalanobis Distance Gate

Uses the Mahalanobis distance (square root of NIS) with a fixed threshold:

$$d_M = \sqrt{v_k^T S_k^{-1} v_k} \leq \text{threshold}$$

```python
from kalbee import ellipsoidal_gate, mahalanobis_distance

# Compute distance
dist = mahalanobis_distance(innovation, innovation_cov)
print(f"Mahalanobis distance: {dist:.3f}")

# Gate with threshold
passed = ellipsoidal_gate(innovation, innovation_cov, gate_threshold=5.0)
```

### Gated Update

High-level function that runs a gated update on any `BaseFilter`:

```python
from kalbee import KalmanFilter, gated_update

kf = KalmanFilter(state, cov, F, Q, H, R)

# Chi-squared gating (default)
updated, state = gated_update(kf, measurement, confidence=0.95)

# Mahalanobis gating
updated, state = gated_update(kf, measurement, gate_threshold=3.0)

if not updated:
    print("Measurement rejected by gate")
```

## When to Use

| ✅ Use gating when | ❌ Don't use when |
|---|---|
| Measurements may contain outliers | All measurements are trusted |
| Multi-target tracking (data association) | Single target, clean measurements |
| Sensor fusion with noisy sensors | — |

---

## Multi-Object Tracking Example

```python
import numpy as np
from kalbee import KalmanFilter, chi2_gate

def associate_and_update(trackers, detections, H, R):
    """Simple nearest-neighbor with gating."""
    updated = []
    for track in trackers:
        innovation = detections - (H @ track.x)
        S = H @ track.P @ H.T + R

        best_det = None
        best_nis = float('inf')

        for i, det in enumerate(detections):
            v = det.reshape(-1, 1) - (H @ track.x)
            nis = float(v.T @ np.linalg.inv(S) @ v)
            _, nis_val, threshold = chi2_gate(v, S)

            if nis_val < threshold and nis < best_nis:
                best_nis = nis
                best_det = i

        if best_det is not None:
            track.predict()
            track.update(detections[best_det].reshape(-1, 1))
            updated.append(track)

    return updated
```
