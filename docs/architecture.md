# Design & Architecture

## Project Structure

```
kalbee/
├── __init__.py                  # Top-level exports
├── py.typed                     # PEP 561 marker — ships type info
├── cli.py                       # `kalbee` CLI (demo/bench/new)
├── models/
│   ├── motion.py                # Motion models (CV, CA, CT, imu_velocity_control)
│   ├── measurement.py           # Measurement models
│   └── attitude.py              # Quaternion attitude EKF cookbook
├── modules/
│   ├── filters/
│   │   ├── base.py              # BaseFilter ABC
│   │   ├── kf_filter.py         # Kalman Filter
│   │   ├── ekf_filter.py        # Extended KF
│   │   ├── ukf_filter.py        # Unscented KF
│   │   ├── sigma_point_ukf.py   # UKF with pluggable sigma points
│   │   ├── sigma_points.py      # Sigma point strategies
│   │   ├── particle_filter.py   # Particle Filter
│   │   ├── enkf_filter.py       # Ensemble KF
│   │   ├── information_filter.py # Information Filter
│   │   ├── abg_filter.py        # Alpha-Beta-Gamma
│   │   ├── adaptive_kf.py       # Adaptive KF
│   │   ├── fading_memory_kf.py  # Fading Memory KF
│   │   ├── hinfinity_filter.py  # H-Infinity Filter
│   │   └── auto_filter.py       # Factory
│   ├── smoothers/
│   │   └── rts_smoother.py      # RTS Smoother
│   ├── learning/
│   │   ├── em.py                # EM parameter learning
│   │   └── auto_tune.py         # NIS-based auto-tuning
│   ├── integration/
│   │   ├── pandas.py / polars.py  # DataFrame integration
│   │   └── sklearn_api.py         # KalmanEstimator (fit/transform/predict)
│   └── utils/
│       ├── metrics.py           # RMSE, NEES, NIS
│       ├── gating.py            # Innovation gating
│       ├── consistency.py       # Formal consistency tests
│       ├── diagnostics.py       # Real-time diagnostics
│       ├── outlier_detector.py  # Outlier detection
│       └── jacobian.py          # Numerical (finite-difference) EKF Jacobians
├── experiments/
│   ├── signals.py               # Signal generators
│   ├── runner.py                # Experiment runner
│   └── results.py               # Results container
└── tests/
    └── test_*.py                # 205 tests
```

## Design Principles

### 1. Common Interface via BaseFilter

Every filter inherits from `BaseFilter` and implements:

```python
class BaseFilter(ABC):
    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray: ...
    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray: ...
    def measure(self, state=None) -> np.ndarray: ...
    def predict_only(self, dt: float = 1.0, **kwargs) -> np.ndarray: ...
    def reset(self, state=None, covariance=None) -> None: ...
    def filter_sequence(self, measurements, dt=1.0, missing=None): ...
    def save_state(self, filepath: str) -> None: ...
    def load_state(self, filepath: str) -> None: ...
```

This means you can swap filters without changing calling code.

### 2. Numerical Stability

- **Joseph form** for covariance updates in KF, EKF, and H-Infinity
- **Symmetry enforcement** after every covariance update: `P = (P + P.T) / 2`
- **Cholesky fallback** in UKF sigma point generation
- **Positive-definite clamping** in auto-tuning to prevent degenerate Q/R

### 3. Strategy Pattern for Sigma Points

The `SigmaPointUKF` delegates sigma point generation to external strategy objects:

```python
# Swap sigma point strategies without changing filter code
ukf_simplex = SigmaPointUKF(..., sigma_points=SimplexSigmaPoints(n))
ukf_merwe = SigmaPointUKF(..., sigma_points=MerweScaledSigmaPoints(n))
ukf_julier = SigmaPointUKF(..., sigma_points=JulierSigmaPoints(n))
```

### 4. Extensibility

Add a new filter by:

1. Create `kalbee/modules/filters/my_filter.py`
2. Inherit from `BaseFilter`
3. Implement `predict()` and `update()`
4. Register in `__init__.py` and `AutoFilter`

```python
from kalbee.modules.filters.base import BaseFilter

class MyFilter(BaseFilter):
    def predict(self, dt=1.0, **kwargs):
        # Your predict logic
        return self.state

    def update(self, measurement, **kwargs):
        # Your update logic
        return self.state
```

## Dependencies

| Package | Why |
|---|---|
| `numpy` | Core matrix operations |
| `scipy` | Cholesky decomposition (UKF) |

## Filter Comparison

| Filter | Linear | Non-linear | Non-Gaussian | Jacobians | Robustness | Complexity |
|---|---|---|---|---|---|---|
| KF | ✅ | ❌ | ❌ | Not needed | Standard | $O(n^3)$ |
| EKF | ✅ | ✅ | ❌ | Required | Standard | $O(n^3)$ |
| UKF | ✅ | ✅ | ❌ | Not needed | Standard | $O(n^3)$ |
| SigmaPointUKF | ✅ | ✅ | ❌ | Not needed | Standard | $O(n^3)$ |
| PF | ✅ | ✅ | ✅ | Not needed | Standard | $O(Nn^2)$ |
| EnKF | ✅ | ✅ | ❌ | Not needed | Standard | $O(Nn^2)$ |
| IF | ✅ | ❌ | ❌ | Not needed | Standard | $O(n^3)$ |
| ABG | ✅ | ❌ | ❌ | Not needed | Standard | $O(1)$ |
| AKF | ✅ | ❌ | ❌ | Not needed | Standard | $O(n^3)$ |
| Fading Memory KF | ✅ | ❌ | ❌ | Not needed | High (discounting) | $O(n^3)$ |
| H-Infinity | ✅ | ❌ | ❌ | Not needed | High (worst-case) | $O(n^3)$ |
| SRKF | ✅ | ❌ | ❌ | Not needed | High (square-root) | $O(n^3)$ |
| IMM | ✅ | ✅ | ❌ | Optional | High (multi-model) | $O(Kn^3)$ |
| VKF | ✅ | ❌ | ❌ | Not needed | Standard | $O(n^3)$ |
