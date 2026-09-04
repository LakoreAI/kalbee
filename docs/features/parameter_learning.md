# Parameter Learning (EM)

Tuning the process- and measurement-noise covariances $Q$ and $R$ by hand is one of the hardest parts of applying a Kalman filter. `em_kalman` learns them **from data** by maximum likelihood, using the Expectation-Maximization algorithm of Shumway & Stoffer (1982).

This is the offline, batch complement to the online [Adaptive Kalman Filter](../filters/adaptive_kalman_filter.md): where the adaptive filter nudges $Q$/$R$ on the fly from the innovation sequence, `em_kalman` fits them to a whole recorded sequence.

## How It Works

Each iteration performs two steps:

- **E-step** — run a Kalman filter forward and an RTS smoother backward over the data (including the lag-one smoothed covariance recursion) to compute the expected sufficient statistics.
- **M-step** — update $Q$ and $R$ in closed form from those statistics.

The marginal log-likelihood is **guaranteed non-decreasing** across iterations:

$$
\mathcal{L} = -\frac{1}{2} \sum_k \left[ m \log(2\pi) + \log|S_k| + v_k^\top S_k^{-1} v_k \right]
$$

## Usage

```python
import numpy as np
from kalbee import em_kalman
from kalbee.models import constant_velocity, position_measurement_model

F, _ = constant_velocity(dt=1.0, n_dims=1)
H, _ = position_measurement_model(order=1, n_dims=1)

# measurements: array of shape (T, m)  (or (T,) for scalar observations)
result = em_kalman(measurements, F, H, n_iter=50)

print("Learned Q:\n", result.Q)
print("Learned R:\n", result.R)
print("Final log-likelihood:", result.loglik_history[-1])
print("Converged:", result.converged, "in", result.n_iter_run, "iterations")
```

## Options

| Parameter | Meaning |
|---|---|
| `Q`, `R` | Initial covariance guesses (default: identity) |
| `x0`, `P0` | Initial state mean/covariance |
| `learn_Q`, `learn_R` | Toggle which covariances are updated |
| `n_iter` | Maximum EM iterations |
| `tol` | Stop when the log-likelihood improves by less than this |

The returned `EMResult` carries `Q`, `R`, the full `loglik_history`, `n_iter_run`, and a `converged` flag.

!!! tip "Fit once, deploy online"
    A common workflow is to fit $Q$/$R$ offline on a representative recording with `em_kalman`, then plug the learned covariances into a live `KalmanFilter`.

!!! warning "What it learns"
    `em_kalman` learns the noise covariances $Q$ and $R$ for a **fixed** structure $F$, $H$. The transition and measurement matrices themselves are not estimated.
