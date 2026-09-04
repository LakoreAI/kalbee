# SigmaPointUKF

The **SigmaPointUKF** is an Unscented Kalman Filter with a pluggable sigma point strategy. Unlike the built-in UKF which hardcodes sigma point generation, this filter delegates sigma point computation to external strategy objects, enabling maximum flexibility and experimentation.

## Fundamental Concepts

### Why Pluggable Sigma Points?

Different sigma point strategies have different numerical properties:

| Strategy | Spread Control | Numerical Stability | Best For |
|---|---|---|---|
| **SimplexSigmaPoints** | $\alpha, \beta, \kappa$ | Standard | General use |
| **MerweScaledSigmaPoints** | $\alpha, \beta, \kappa$ | Excellent for large $n$ | High-dimensional systems |
| **JulierSigmaPoints** | $\kappa$ | Robust | Certain non-Gaussian systems |

### The Algorithm

The UKF uses the **unscented transform** to propagate statistics through nonlinear functions:

1. Generate $2n+1$ sigma points from $(x, P)$
2. Propagate each point through $f$ (predict) or $h$ (update)
3. Reconstruct mean and covariance from transformed points

### When to Use

| ✅ Use SigmaPointUKF when | ❌ Don't use when |
|---|---|
| You need to experiment with sigma point strategies | Standard UKF works fine |
| High-dimensional state space | Low-dimensional systems (use standard UKF) |
| Custom sigma point weighting needed | Linear system (use KF) |
| Research on UKF variants | — |

---

## How to Use

### Basic Example

```python
import numpy as np
from kalbee import SigmaPointUKF, MerweScaledSigmaPoints

state = np.zeros((2, 1))
covariance = np.eye(2) * 10.0
Q = np.eye(2) * 0.01
R = np.array([[0.5]])

# Define nonlinear functions
def transition(x, dt):
    return np.array([
        [x[0, 0] + x[1, 0] * dt],
        [x[1, 0]]
    ])

def measurement(x):
    return np.array([[x[0, 0]]])

# Create UKF with MerweScaled sigma points
sigma_pts = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0.0)

ukf = SigmaPointUKF(
    state, covariance, Q, R,
    transition_function=transition,
    measurement_function=measurement,
    sigma_points=sigma_pts
)

measurements = [1.2, 2.1, 2.8, 4.1, 5.0]
for z in measurements:
    ukf.predict(dt=1.0)
    ukf.update(np.array([[z]]))
    print(f"Position: {ukf.x[0,0]:.2f}, Velocity: {ukf.x[1,0]:.2f}")
```

### Swapping Sigma Point Strategies

```python
from kalbee import SimplexSigmaPoints, JulierSigmaPoints

# Simplex (default)
ukf_simplex = SigmaPointUKF(
    state, cov, Q, R, f, h,
    sigma_points=SimplexSigmaPoints(n=2)
)

# Julier
ukf_julier = SigmaPointUKF(
    state, cov, Q, R, f, h,
    sigma_points=JulierSigmaPoints(n=2, kappa=0.0)
)

# Auto-create from parameters (uses Simplex by default)
ukf_auto = SigmaPointUKF(
    state, cov, Q, R, f, h,
    alpha=0.001, beta=2.0, kappa=0.0
)
```

---

## Sigma Point Strategies

### SimplexSigmaPoints

Standard UKF sigma points. Produces $2n+1$ points symmetrically around the mean.

```python
from kalbee import SimplexSigmaPoints

sp = SimplexSigmaPoints(n=4, alpha=0.001, beta=2.0, kappa=0.0)
sigma_points = sp.sigma_points(x, P)  # (2n+1, n) array
```

### MerweScaledSigmaPoints

Scalable sigma points with better numerical properties for large state dimensions.

```python
from kalbee import MerweScaledSigmaPoints

sp = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0.0)
sigma_points = sp.sigma_points(x, P)
```

### JulierSigmaPoints

Original Julier formulation. More robust for certain non-Gaussian applications.

```python
from kalbee import JulierSigmaPoints

sp = JulierSigmaPoints(n=4, kappa=0.0)
sigma_points = sp.sigma_points(x, P)
```

---

## Run an Experiment

```python
from kalbee import run_experiment

report = run_experiment(
    signal="sine",
    filters=["ukf", "sigma_point_ukf"],
    noise_std=0.5,
    duration=10.0,
    seed=42,
)
print(report.summary())
```
