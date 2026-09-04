# kalbee

<div align="center">
  <img src="https://raw.githubusercontent.com/MinLee0210/kalbee/main/docs/kalbee.png" alt="kalbee logo" width="250"/>
</div>

<br>

**kalbee** is a clean, modular Python library for **Kalman Filters and state estimation algorithms**. It provides a unified interface for 15 different filter types, smoothers, diagnostic metrics, and a built-in experiment runner to compare filter performance.

## Highlights

| Category | What you get |
|---|---|
| **15 Filters** | KF, EKF, UKF, Particle Filter, Ensemble, Information, Alpha-Beta-Gamma, Adaptive KF, Square-Root KF, Vectorized KF, Fading Memory KF, H-Infinity, SigmaPointUKF, IMM |
| **Sigma Points** | Simplex, MerweScaled, Julier — pluggable via Strategy pattern |
| **Motion Models** | Ready-made constant-velocity, constant-acceleration, and coordinated-turn $(F, Q)$ builders |
| **Tracking** | SORT-style multi-object tracker with Hungarian association and gating |
| **Learning** | Offline EM to fit $Q$/$R$ from data, plus NIS-based auto-tuning |
| **Smoother** | Rauch-Tung-Striebel (RTS) backward smoother |
| **Diagnostics** | RMSE, NEES, NIS, Log-Likelihood, FilterDiagnostics, consistency tests |
| **Outlier Rejection** | Chi-squared gating, Mahalanobis gating, adaptive outlier detection |
| **Experiments** | One-liner to compare filters on synthetic signals |
| **Stability** | Joseph form covariance updates, Cholesky factor stabilization, symmetry checks |
| **Utilities** | Batch processing, state serialization, control inputs, missing data handling |

## Quick Start

```bash
pip install kalbee
```

```python
from kalbee import run_experiment

# Compare filters on a sine wave
report = run_experiment(
    signal="sine",
    filters=["kf", "ekf", "ukf", "pf"],
    noise_std=0.5,
)
print(report.summary())
```

## Navigation

- **[Getting Started](getting_started.md)** — Installation, core concepts, first filter
- **[Filters](filters/kalman_filter.md)** — Deep dive into each filter with theory + code
- **[Features](features/motion_models.md)** — Motion models, tracking, learning, diagnostics
- **[Architecture](architecture.md)** — Design philosophy and extensibility
