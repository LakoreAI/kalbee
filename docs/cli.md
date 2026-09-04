# Command-Line Interface

The `kalbee` CLI is a zero-code way to see the library work — no Python
script required. Core commands (`demo`, `bench`) need nothing beyond
kalbee's normal numpy/scipy dependencies; the animated `--live` view needs
`rich`: `pip install kalbee[cli]`.

## `kalbee demo`

Runs a filter against a synthetic signal and reports tracking accuracy.

```bash
kalbee demo --filter akf --signal maneuver --steps 200
```

Add `--live` for an animated terminal chart (true signal, noisy measurement,
and filtered estimate, each as a live-updating sparkline):

```bash
kalbee demo --live --filter kf --signal sine
```

| Flag | Default | Meaning |
|---|---|---|
| `--filter` | `kf` | One of `kf`, `akf`, `srkf`, `if`, `vkf`, `fmkf`, `hinf` (filters that share the plain `(F, Q, H, R)` constructor this demo builds). |
| `--signal` | `sine` | `sine`, `cosine`, `linear`, `step`, or `maneuver`. |
| `--steps` | `150` | Number of time steps. |
| `--noise` | `0.3` | Measurement noise std. |
| `--process-var` | `0.1` | Process-noise variance. |
| `--live` | off | Animate in the terminal (needs `kalbee[cli]`). |
| `--fps` | `15` | Frames per second in `--live` mode. |

## `kalbee bench`

Runs the built-in filter benchmark suite (speed + accuracy across all core
filters) and prints a sorted comparison table.

```bash
kalbee bench --duration 10 --dt 0.05
```

For a comparison against FilterPy/pykalman/simdkalman on identical data, see
[Benchmarks](benchmarks.md) and `scripts/compare_benchmarks.py`.

## `kalbee new`

Scaffolds a runnable starter script so you don't start from a blank file.

```bash
kalbee new my_tracker.py                       # basic 1-D KF tracking loop
kalbee new gps_fusion.py --template gps_imu     # GPS + IMU recipe
kalbee new attitude.py --template attitude      # quaternion attitude EKF recipe
```

Pass `--force` to overwrite an existing file.
