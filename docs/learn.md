# Learn: Kalman Filtering, Intuition First

kalbee is a library with **one idea** behind every filter:

> Guess where the system is. *Predict.*
> Look at the measurement. *Update.*
> Blend the two by how much you trust each. Repeat.

This page builds that intuition with numbers first and notation second, maps
the math to kalbee's arguments, and ends with a learning path. It follows the
pedagogy of Roger Labbe's free
[*Kalman and Bayesian Filters in Python*](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
and the worked examples of [kalmanfilter.net](https://kalmanfilter.net).

---

## 1. Start with one number

Suppose we estimate the height of a building. It does not move, so our model
of the world is "constant". We have a noisy altimeter whose standard
deviation is known: 5 m (variance $r = 25$).

A measurement $z_1 = 49.03$ arrives. Before it, our best guess was
$\hat{x} = 60$ with uncertainty $p = 225$ (an educated guess, ±15 m).

How much should the new measurement move our guess? A *weighted average*:

$$\hat{x}_\text{new} = (1 - K)\,\hat{x} + K\,z,$$

where the **Kalman gain** $K$ is the ratio of *our* uncertainty to the *total*
uncertainty:

$$K = \frac{p}{p + r} = \frac{225}{225 + 25} = 0.9.$$

Because our guess was 9× more uncertain than the measurement, we trust the
measurement 90%. And the uncertainty of the blend shrinks to

$$p_\text{new} = (1 - K)\,p = (1 - 0.9)\,225 = 22.5.$$

Next measurement arrives, and we repeat with $p = 22.5$. Now $K = 22.5 /
(22.5 + 25) \approx 0.47$ — we already have a decent estimate, so the noisy
sensor matters less. The gain *automatically* decreases as the estimate
improves. That self-tuning weight is the whole magic of a Kalman filter.

### The same thing with kalbee

```python
import numpy as np
from kalbee import KalmanFilter

measurements = [49.03, 48.44, 55.21, 49.98, 50.6, 52.61, 45.87, 42.64, 48.26, 55.84]

# State [height]; F=1 means "constant"; H=1 means "we measure height directly"
kf = KalmanFilter(
    state=np.array([[60.0]]),          # x_0   : initial guess
    covariance=np.array([[225.0]]),    # P_0   : how sure we are (±15 m)
    transition_matrix=np.array([[1.0]]),          # F
    transition_covariance=np.array([[0.0]]),      # Q: building doesn't move
    measurement_matrix=np.array([[1.0]]),         # H
    measurement_covariance=np.array([[25.0]]),    # R: altimeter variance
)

for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))
    print(f"estimate={kf.x[0,0]:6.2f} m   uncertainty=±{kf.P[0,0]**0.5:4.2f} m")
```

The printed estimates converge to the true 50 m, and the uncertainty falls
from ±15 m to about ±1.6 m. This matches the hand table at
[kalmanfilter.net/example](https://kalmanfilter.net/kalman1d.html) exactly —
kalbee is the same arithmetic, just batched into matrices.

---

## 2. The whole filter in five lines

For multiple variables everything becomes matrix multiplication, but the
shape is identical:

| # | kalbee | What it does |
|---|--------|--------------|
| 1 | `kf.predict()` | push the state forward with physics: $x = Fx$, $P = FPF^T + Q$ |
| 2 | `y = z - H @ x` | the *innovation*: "how wrong was my prediction, in sensor units?" |
| 3 | `S = H @ P @ H.T + R` | the uncertainty of that prediction, in sensor units |
| 4 | `K = P @ H.T @ inv(S)` | the gain: blend by relative uncertainty (matrix version of §1) |
| 5 | `kf.update(z)` | $x = x + Ky$, $P = (I-KH)P$ — trust the better source more |

Watch these five lines do the work in
[the denoising animation](examples.md) (`assets/gif/filter_demo.gif`): the
blue band is the filter *telling you* how uncertain it is, and it shrinks as
measurements accumulate.

---

## 3. Translating the math to kalbee

Every symbol in the recursion maps to one constructor argument:

| Symbol | Meaning | kalbee | Where it comes from |
|--------|---------|--------|---------------------|
| $x_0$, $P_0$ | initial guess + uncertainty | `state`, `covariance` | your best guess; use a large $P_0$ if unsure |
| $F$ | how the state evolves | `transition_matrix` | `kalbee.models.constant_velocity(dt, ...)` etc. |
| $Q$ | how much the *model* lies (process noise) | `transition_covariance` | the tuning lever (see §4) |
| $H$ | state → measurement | `measurement_matrix` | `kalbee.models.position_measurement_model(...)` |
| $R$ | sensor noise | `measurement_covariance` | vendor spec, or calibrate |
| $z$ | measurement | `update(z)` | your data |
| $K$ | trust ratio | (computed inside `update`) | not stored; `last_y`/`last_S` carry the innovation |
| $y$, $S$ | innovation + covariance | (internal) | `kf.last_y`, `kf.last_S` — your diagnostics |

Two rules of thumb:

* State layout is per-axis blocks: for position+velocity on $(x, y)$ the
  state is `[x, vx, y, vy]`. The model builders follow this convention, so
  just tell them how many axes you have (`n_dims`).
* If your state *is not* what you measure, `H` maps between them. kalbee's
  `position_measurement_model` only "sees" the position entries of each axis.

---

## 4. The two levers: $Q$ and $R$

Almost all filter tuning reduces to two questions.

* **$Q$ too large** → the filter distrusts its model, chases every
  measurement, is *noisy* (and underconfident).
* **$Q$ too small** → the filter is overconfident in its model, lags real
  changes, and its uncertainty band is *wrong*.
* **$R$ too large** → slow to react; smooth but laggy.
* **$R$ too small** → jittery; trusts a possibly-bad sensor too much.

So how do you know *without ground truth* whether you over- or under-tuned?
Use consistency statistics. For a correctly tuned filter the *normalized
innovation squared* should average to the measurement dimension:

```python
from kalbee import KalmanFilter, FilterDiagnostics
from kalbee.models import constant_velocity, position_measurement_model

F, Q = constant_velocity(dt=dt, process_var=0.1, n_dims=1)
H, R = position_measurement_model(order=1, n_dims=1, measurement_var=1.0)
kf = KalmanFilter(np.zeros((2, 1)), np.eye(2) * 100, F, Q, H, R)
diag = FilterDiagnostics(m=1, n=2)     # m = measurement dim, n = state dim

for z in stream:
    kf.predict(); kf.update(z)
    diag.collect(kf)                     # needs no ground truth
print(diag.summary()["nis_mean"])        # want ≈ m
```

`nis_mean ≫ m` means the filter is overconfident (raise $Q$ or $R$);
`nis_mean ≪ m` means underconfident. kalbee can even do the loop for you:
`tune_kalman_filter`, `quick_tune`, or the offline `em_kalman` (see
[Parameter Learning](features/parameter_learning.md) and
[Auto-Tuning](features/auto_tuning.md)). With ground truth available
(simulation, datasets), `NEES` checks the same for the state, and the
innovation *whiteness* test catches model mismatch.

---

## 5. From one filter to everything else

kalbee's filters are all the recursion of §2 with a different way of
representing the probability distribution:

| Representation | Filter | Use when |
|---|---|---|
| Gaussian moments, linear | `KalmanFilter` | the default |
| Gaussian, linearized | `ExtendedKalmanFilter` | mild nonlinearities |
| Gaussian, sigma points | `UnscentedKalmanFilter` | nonlinear, no Jacobians |
| Particle set | `ParticleFilter` | non-Gaussian, multimodal |
| Several models blended | `InteractingMultipleModel` | maneuvering targets |
| Many objects | `MultiObjectTracker` | §2 + association (see the [MOT16 demo](examples.md)) |

Before reaching for a "better" filter, ask whether your *model*
(`F`, `Q`, `H`, `R`) is the problem. A wrong model beats any filter, and a
consistency check (§4) will tell you immediately.

---

## Suggested path

1. **[Getting Started](getting_started.md)** — install and your first loop.
2. **This page** — rebuild §1 by hand once, with pen and paper.
3. **[Examples & Gallery](examples.md)** — watch the filters work, then run
   `scripts/generate_demo_gif.py` yourself.
4. **[Filtering & design notes](filtering_logic.md)** — why covariance updates
   use the Joseph form.
5. Work one doc per filter: [Kalman](filters/kalman_filter.md) → [EKF](filters/extended_kalman_filter.md) → [UKF](filters/unscented_kalman_filter.md) → [IMM](filters/interacting_multiple_model.md) → [PF](filters/particle_filter.md).
6. Try the exercises in `examples/`, then compare answers with the [MOT16](features/yolo_tracking.md) and [GPS+IMU](features/sensor_fusion_cookbook.md) recipes.

### Deeper (free) resources

- **Roger Labbe — *Kalman and Bayesian Filters in Python***: the intuition-first
  book, written as runnable notebooks.
  [github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- **kalmanfilter.net**: worked 1-D examples and figures behind every equation.
  [kalmanfilter.net](https://kalmanfilter.net)
- **Bar-Shalom, Li & Kirubarajan — *Estimation with Applications to Tracking
  and Navigation*** (Wiley): the standard reference for data association and
  multi-target tracking.
- **Thrun, Burgard & Fox — *Probabilistic Robotics*** (MIT Press): filtering
  in the context of robotics and mapping.
