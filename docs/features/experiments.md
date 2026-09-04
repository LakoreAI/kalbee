# Experiment Runner

The experiment runner lets you **compare multiple filters** on synthetic signals with a single function call. It handles signal generation, filter setup, execution, and performance reporting.

## Quick Start

```python
from kalbee import run_experiment

report = run_experiment(
    signal="sine",
    filters=["kf", "ekf", "ukf", "pf", "enkf", "if", "akf"],
    noise_std=0.5,
    duration=10.0,
    seed=42,
)
print(report.summary())
```

## Available Signals

| Signal | Description | Best for testing |
|---|---|---|
| `"sine"` | Sinusoidal oscillation | Non-linear dynamics tracking |
| `"cosine"` | Cosine wave | Phase-shifted oscillation |
| `"linear"` | Constant velocity | KF optimality check |
| `"step"` | Sudden jump | Filter responsiveness |

### Signal Parameters

```python
report = run_experiment(
    signal="sine",
    filters=["kf"],
    duration=20.0,        # Length in seconds
    dt=0.05,              # Time step (smaller = more samples)
    noise_std=1.0,        # Measurement noise level
    process_noise=0.1,    # Process noise intensity
    seed=42,              # Reproducibility
    signal_kwargs={       # Pass to signal generator
        "amplitude": 2.0,
        "frequency": 0.3,
    },
)
```

### Custom Signals

```python
import numpy as np
from kalbee.experiments import custom_signal

# Any function you want
t, states, measurements = custom_signal(
    func=lambda t: np.sin(t) + 0.5 * np.cos(3 * t),
    derivative=lambda t: np.cos(t) - 1.5 * np.sin(3 * t),
    duration=10.0,
    noise_std=0.3,
    seed=42,
)
```

## Available Filters

| Short name | Filter | Notes |
|---|---|---|
| `"kf"` | Kalman Filter | Linear baseline |
| `"ekf"` | Extended Kalman Filter | Non-linear with Jacobians |
| `"ukf"` | Unscented Kalman Filter | Non-linear without Jacobians |
| `"pf"` | Particle Filter | 500 particles |
| `"enkf"` | Ensemble Kalman Filter | 100 ensemble members |
| `"if"` | Information Filter | Dual of KF |
| `"akf"` | Adaptive Kalman Filter | Online Q/R estimation |

## Working with Results

### Report Summary

```python
print(report.summary())
```

```
================================================================
  Experiment Report: sine
================================================================
  Signal params: duration=10.0, dt=0.1, noise_std=0.5

  Filter                      Pos RMSE   Vel RMSE   Avg NEES
  -------------------------------------------------------
  KF                            0.1234     0.5678     2.1000
  EKF                           0.1245     0.5690     2.1200
  ...
================================================================
  🏆 Best position tracking: KF
```

### Individual Results

```python
for result in report.results:
    print(f"{result.filter_name}:")
    print(f"  Position RMSE: {result.position_rmse():.4f}")
    print(f"  Velocity RMSE: {result.velocity_rmse():.4f}")
    print(f"  Average NEES:  {result.average_nees():.2f}")
```

### Best Filter

```python
best = report.get_best_filter()
print(f"Best filter: {best.filter_name}")
```

### Machine-Readable Output

```python
data = report.to_dict()
# {'signal': 'sine', 'params': {...}, 'results': [...], 'best_filter': 'KF'}
```

### Access Raw Data

```python
result = report.results[0]  # First filter's result

result.time_steps          # (T,) array of time values
result.true_states         # (T, 2, 1) ground truth
result.estimated_states    # (T, 2, 1) filter estimates
result.measurements        # (T, 1, 1) noisy measurements
result.covariances         # List of (2, 2) covariance matrices
```

## Example: Comparing on Different Signals

```python
from kalbee import run_experiment

signals = ["sine", "cosine", "linear", "step"]
filters = ["kf", "ekf", "ukf"]

for sig in signals:
    report = run_experiment(signal=sig, filters=filters, seed=42)
    best = report.get_best_filter()
    print(f"{sig:>8s}: Best = {best.filter_name}, "
          f"RMSE = {best.position_rmse():.4f}")
```
