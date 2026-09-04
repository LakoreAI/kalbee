# scikit-learn Integration

`KalmanEstimator` wraps any kalbee filter behind the `fit`/`transform`/
`predict` convention, so it drops straight into an `sklearn.pipeline.Pipeline`
or a `GridSearchCV` sweep. Requires scikit-learn: `pip install kalbee[sklearn]`.

## One-liner smoothing

```python
import numpy as np
from kalbee.modules.integration.sklearn_api import KalmanEstimator

t = np.linspace(0, 10, 200)
noisy = np.sin(t) + np.random.normal(0, 0.2, 200)

smoothed = KalmanEstimator(dt=t[1] - t[0], process_var=5.0, measurement_var=0.2).fit_transform(noisy)
```

Each column of `X` is one measured spatial axis (pass a 2-D array like
`[[x, y], ...]` for 2-D position tracking); each row is one time step.

## Inside a Pipeline

```python
from sklearn.pipeline import Pipeline
from kalbee.modules.integration.sklearn_api import KalmanEstimator

pipe = Pipeline([
    ("kalman", KalmanEstimator(order=2, process_var=1.0, measurement_var=0.5)),
    # ... downstream estimator, e.g. a classifier on the smoothed trajectory
])
smoothed = pipe.fit_transform(raw_measurements)
```

## Parameters

| Parameter | Meaning |
|---|---|
| `mode` | Any [`AutoFilter`](../architecture.md) mode (`"kf"`, `"akf"`, `"srkf"`, ...). |
| `order` | Kinematic order — 1 = constant-velocity, 2 = constant-acceleration. |
| `dt` | Time step between rows of `X`. |
| `process_var` / `measurement_var` | Noise variances for the default motion/measurement model. |
| `tune` | If `True`, auto-tune `Q`/`R` from `X` via [`quick_tune`](auto_tuning.md) instead. |
| `return_full_state` | If `True`, `transform` returns the full state vector (e.g. position *and* velocity) instead of just position. |

Each `transform()` call re-runs the filter from its initial state, so
repeated calls (including sklearn's own `fit_transform`/`predict` calls) are
independent — no carry-over state between calls.
