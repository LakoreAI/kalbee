# kalbee

<div align="center">
  <img src="https://raw.githubusercontent.com/LakoreAI/kalbee/main/docs/kalbee.png" alt="kalbee logo" width="250"/>
</div>

<br>

**kalbee** is a clean, modular Python **toolkit for filtering and tracking**. It provides a unified interface for 18 filter types, smoothers, multi-object trackers, diagnostic metrics, and a built-in experiment runner to compare filter performance.

## Highlights

| Category | What you get |
|---|---|
| **18 Filters** | KF, EKF, UKF, SigmaPointUKF, CKF, PF, RBPF, EnKF, Information, ABG, Adaptive, Square-Root/Cholesky, Vectorized, Fading-Memory, H∞, IMM, VB, InEKF |
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

## See It In Action

<figure>
  <img src="assets/gif/filter_demo.gif" alt="Kalman filter smoothing a noisy signal" width="600"/>
  <figcaption>A Kalman filter turning noisy measurements into a clean position
  and velocity estimate, with the ±1σ uncertainty band shrinking as
  measurements arrive.</figcaption>
</figure>

Animated demos for filtering, maneuvering targets (IMM) and real-pedestrian
multi-object tracking live in the **[Examples & Gallery](examples.md)**.

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
- **[Examples & Gallery](examples.md)** — Animated demos and copy-paste recipes
- **[Filters](filters/kalman_filter.md)** — Deep dive into each filter with theory + code
- **[Features](features/motion_models.md)** — Motion models, tracking, learning, diagnostics
- **[Architecture](architecture.md)** — Design philosophy and extensibility
