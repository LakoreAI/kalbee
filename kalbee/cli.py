"""
``kalbee`` command-line interface.

Zero-code ways to see the library work:

    kalbee demo --live          # animated terminal chart of a filter tracking noise
    kalbee bench                # speed/accuracy comparison across all filters
    kalbee new my_tracker.py    # scaffold a starter script

Core commands (``demo``, ``bench``) only need numpy/scipy, same as the rest
of kalbee. ``--live`` additionally needs ``rich``: ``pip install kalbee[cli]``.
"""

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from kalbee import __version__

_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int = 60) -> str:
    """Render a list of numbers as a one-line Unicode block sparkline."""
    if not values:
        return ""
    window = values[-width:]
    lo, hi = min(window), max(window)
    span = hi - lo
    if span < 1e-9:
        return _BLOCKS[0] * len(window)
    return "".join(_BLOCKS[int((v - lo) / span * (len(_BLOCKS) - 1))] for v in window)


def _build_filter(mode: str, dt: float, process_var: float, measurement_var: float):
    from kalbee.modules.filters.auto_filter import AutoFilter
    from kalbee.models import constant_velocity, position_measurement_model

    F, Q = constant_velocity(dt=dt, process_var=process_var, n_dims=1)
    H, R = position_measurement_model(
        order=1, n_dims=1, measurement_var=measurement_var
    )
    x0 = np.zeros((2, 1))
    P0 = np.eye(2) * 100.0
    return AutoFilter.from_filter(x0, P0, F, Q, H, R, mode=mode)


def cmd_demo(args: argparse.Namespace) -> int:
    from kalbee.experiments.signals import SIGNALS
    from kalbee.modules.utils.metrics import rmse as rmse_fn

    signal_fn = SIGNALS.get(args.signal)
    if signal_fn is None:
        print(f"Unknown signal '{args.signal}'. Available: {', '.join(SIGNALS)}")
        return 1

    dt = 0.1
    duration = args.steps * dt
    t, true_states, measurements = signal_fn(  # type: ignore[operator]
        duration=duration, dt=dt, noise_std=args.noise, seed=args.seed
    )

    try:
        kf = _build_filter(args.filter, dt, args.process_var, args.noise**2)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.live:
        return _run_live_demo(kf, t, true_states, measurements, args, rmse_fn)
    return _run_static_demo(kf, t, true_states, measurements, args, rmse_fn)


def _run_static_demo(kf, t, true_states, measurements, args, rmse_fn) -> int:
    estimates = []
    for i in range(len(t)):
        kf.predict(dt=0.1)
        kf.update(measurements[i])
        estimates.append(kf.x[0, 0])

    est = np.array(estimates)
    true_pos = true_states[:, 0, 0]
    meas_pos = measurements[:, 0, 0]

    print(f"kalbee demo — filter={args.filter} signal={args.signal} steps={len(t)}")
    print(f"  measurement RMSE: {rmse_fn(meas_pos, true_pos):.4f}")
    print(f"  filtered RMSE:    {rmse_fn(est, true_pos):.4f}")
    print(
        "\nTip: add --live for an animated terminal view (needs `pip install kalbee[cli]`)."
    )
    return 0


def _run_live_demo(kf, t, true_states, measurements, args, rmse_fn) -> int:
    try:
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print(
            "Live mode needs `rich`. Install it with: pip install kalbee[cli]\n"
            "Falling back to the static summary:\n"
        )
        return _run_static_demo(kf, t, true_states, measurements, args, rmse_fn)

    true_hist: List[float] = []
    meas_hist: List[float] = []
    est_hist: List[float] = []

    def render():
        lines = [
            Text.assemble(("true      ", "dim"), (_sparkline(true_hist), "dim white")),
            Text.assemble(("measured  ", "yellow"), (_sparkline(meas_hist), "yellow")),
            Text.assemble(
                ("filtered  ", "bold green"), (_sparkline(est_hist), "bold green")
            ),
        ]
        stats = ""
        if len(est_hist) > 1:
            stats = (
                f"step {len(est_hist)}/{len(t)}  "
                f"filtered RMSE so far: {rmse_fn(np.array(est_hist), np.array(true_hist)):.4f}"
            )
        body = Group(*lines, Text(stats, style="dim"))
        return Panel(
            body,
            title=f"kalbee demo — {args.filter} on {args.signal}",
            subtitle="Ctrl+C to stop",
        )

    with Live(render(), refresh_per_second=args.fps, screen=False) as live:
        try:
            for i in range(len(t)):
                kf.predict(dt=0.1)
                kf.update(measurements[i])

                true_hist.append(true_states[i, 0, 0])
                meas_hist.append(measurements[i, 0, 0])
                est_hist.append(kf.x[0, 0])

                live.update(render())
                time.sleep(1.0 / args.fps)
        except KeyboardInterrupt:
            pass

    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from kalbee.experiments.benchmark import run_benchmark

    results = run_benchmark(
        duration=args.duration,
        dt=args.dt,
        noise_std=args.noise,
        process_noise=args.process_var,
    )

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="kalbee filter benchmark (sorted by accuracy)")
        table.add_column("Filter")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Position RMSE", justify="right")
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
            table.add_row(name, f"{r['time_ms']:.2f}", f"{r['rmse']:.4f}")
        Console().print(table)
    except ImportError:
        pass  # run_benchmark() already printed a plain-text table.

    return 0


_TEMPLATES = {
    "tracking": '''"""Starter script: track a noisy 1-D signal with a Kalman filter."""

import numpy as np
from kalbee import KalmanFilter
from kalbee.models import constant_velocity, position_measurement_model

dt = 0.1
F, Q = constant_velocity(dt=dt, process_var=0.1, n_dims=1)
H, R = position_measurement_model(order=1, n_dims=1, measurement_var=0.25)

kf = KalmanFilter(np.zeros((2, 1)), np.eye(2) * 100.0, F, Q, H, R)

# Replace with your own data (iterable of scalars or (1, 1) arrays). This
# synthetic noisy sine wave is here so the script runs out of the box.
measurements = np.sin(np.arange(50) * dt) + np.random.normal(0, 0.3, 50)

for z in measurements:
    kf.predict(dt=dt)
    kf.update(np.array([[z]]))
    print(kf.x[0, 0])  # filtered position estimate
''',
    "gps_imu": '''"""Starter script: loosely-coupled GPS+IMU fusion. See examples/gps_imu_fusion.py."""

import numpy as np
from kalbee import KalmanFilter
from kalbee.models import constant_velocity, imu_velocity_control, position_measurement_model

n_dims = 2
dt_imu = 0.02
F, Q = constant_velocity(dt=dt_imu, process_var=0.02, n_dims=n_dims)
B = imu_velocity_control(dt=dt_imu, n_dims=n_dims)
H, R = position_measurement_model(order=1, n_dims=n_dims, measurement_var=1.5**2)

kf = KalmanFilter(np.zeros((4, 1)), np.eye(4) * 100.0, F, Q, H, R, control_matrix=B)

for imu_tick in range(1000):
    accel_reading = np.array([0.0, 0.0])  # your (gravity-compensated) accelerometer, per tick
    kf.predict(u=accel_reading)

    if imu_tick % 25 == 0:
        gps_fix = np.array([[0.0], [0.0]])  # your GPS position fix, every ~25 ticks
        kf.update(gps_fix)

    print(kf.state[[0, 2], 0])  # [x, y] estimate
''',
    "attitude": '''"""Starter script: quaternion attitude EKF. See examples/quaternion_attitude_ekf.py."""

import numpy as np
from kalbee import ExtendedKalmanFilter
from kalbee.models import (
    quaternion_normalize, attitude_transition, attitude_transition_jacobian,
    gravity_measurement, gravity_measurement_jacobian,
)

dt = 0.01
ekf = ExtendedKalmanFilter(
    state=np.array([[1.0], [0.0], [0.0], [0.0]]),
    covariance=np.eye(4) * 0.1,
    transition_covariance=np.eye(4) * 1e-4,
    measurement_covariance=np.eye(3) * 0.05,  # tuning knob, see attitude.py docstring
)

for tick in range(1000):
    gyro = np.array([0.0, 0.0, 0.0])  # your gyroscope reading (rad/s), per tick
    ekf.predict(
        dt=dt,
        f=lambda x, dt: attitude_transition(x, dt, gyro),
        F=lambda x, dt: attitude_transition_jacobian(x, dt, gyro),
    )
    ekf.state = quaternion_normalize(ekf.state)

    accel = np.array([[0.0], [0.0], [1.0]])  # your (unit-normalized) accelerometer reading
    ekf.update(accel, h=gravity_measurement, H=gravity_measurement_jacobian)
    ekf.state = quaternion_normalize(ekf.state)

    print(ekf.state.flatten())  # [w, x, y, z] orientation estimate
''',
}


def cmd_new(args: argparse.Namespace) -> int:
    import os

    path = args.name if args.name.endswith(".py") else f"{args.name}.py"
    if os.path.exists(path) and not args.force:
        print(f"'{path}' already exists. Use --force to overwrite.")
        return 1

    with open(path, "w") as f:
        f.write(_TEMPLATES[args.template])

    print(f"Wrote {path} (template: {args.template})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kalbee", description="kalbee — Kalman filters for humans."
    )
    parser.add_argument("--version", action="version", version=f"kalbee {__version__}")
    sub = parser.add_subparsers(dest="command")

    demo_p = sub.add_parser("demo", help="Run a filter against a synthetic signal.")
    demo_p.add_argument(
        "--filter",
        default="kf",
        choices=["kf", "akf", "srkf", "if", "vkf", "fmkf", "hinf"],
        help="Filter to run (linear filters only — this demo builds a plain (F, Q, H, R) model).",
    )
    demo_p.add_argument(
        "--signal",
        default="sine",
        help="Signal name (sine, cosine, linear, step, maneuver).",
    )
    demo_p.add_argument("--steps", type=int, default=150, help="Number of time steps.")
    demo_p.add_argument(
        "--noise", type=float, default=0.3, help="Measurement noise std."
    )
    demo_p.add_argument("--process-var", dest="process_var", type=float, default=0.1)
    demo_p.add_argument("--seed", type=int, default=42)
    demo_p.add_argument(
        "--live", action="store_true", help="Animate an ASCII chart in the terminal."
    )
    demo_p.add_argument("--fps", type=float, default=15.0)
    demo_p.set_defaults(func=cmd_demo)

    bench_p = sub.add_parser(
        "bench", help="Benchmark speed/accuracy across all filters."
    )
    bench_p.add_argument("--duration", type=float, default=10.0)
    bench_p.add_argument("--dt", type=float, default=0.05)
    bench_p.add_argument("--noise", type=float, default=0.3)
    bench_p.add_argument("--process-var", dest="process_var", type=float, default=0.1)
    bench_p.set_defaults(func=cmd_bench)

    new_p = sub.add_parser("new", help="Scaffold a starter script.")
    new_p.add_argument("name", help="Output filename (.py appended if missing).")
    new_p.add_argument("--template", choices=sorted(_TEMPLATES), default="tracking")
    new_p.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    new_p.set_defaults(func=cmd_new)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
