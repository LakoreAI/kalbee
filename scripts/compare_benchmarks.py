"""
Head-to-head speed/accuracy comparison of kalbee vs. FilterPy, pykalman, and
simdkalman on an identical constant-velocity tracking task.

Not part of the installed package — a reproducibility script for the numbers
quoted in README.md / docs/benchmarks.md. Needs the ``benchmark`` dependency
group: ``uv sync --group benchmark`` (or ``pip install filterpy pykalman
simdkalman``).

Run: python scripts/compare_benchmarks.py
"""

import time
import numpy as np

from kalbee import KalmanFilter, VectorizedKalmanFilter
from kalbee.experiments.signals import sine_signal
from kalbee.modules.utils.metrics import rmse


def _task(seed=0):
    """Shared constant-velocity tracking task: noisy sine, dt=0.05, 10s."""
    dt = 0.05
    t, true_states, measurements = sine_signal(
        duration=10.0, dt=dt, noise_std=0.3, seed=seed
    )
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * 0.1
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.09]])
    x0 = np.zeros((2, 1))
    P0 = np.eye(2) * 100.0
    return dt, true_states, measurements, F, Q, H, R, x0, P0


def bench_kalbee(reps=20):
    dt, true_states, measurements, F, Q, H, R, x0, P0 = _task()

    start = time.perf_counter()
    for _ in range(reps):
        kf = KalmanFilter(x0.copy(), P0.copy(), F, Q, H, R)
        est = np.zeros((len(measurements), 1))
        for i in range(len(measurements)):
            kf.predict()
            kf.update(measurements[i])
            est[i] = kf.x[0, 0]
    elapsed_ms = (time.perf_counter() - start) / reps * 1000.0

    return elapsed_ms, rmse(est[:, 0], true_states[:, 0, 0])


def bench_filterpy(reps=20):
    try:
        from filterpy.kalman import KalmanFilter as FPKalmanFilter
    except ImportError:
        return None

    dt, true_states, measurements, F, Q, H, R, x0, P0 = _task()

    start = time.perf_counter()
    for _ in range(reps):
        kf = FPKalmanFilter(dim_x=2, dim_z=1)
        kf.x = x0.copy()
        kf.P = P0.copy()
        kf.F = F
        kf.Q = Q
        kf.H = H
        kf.R = R
        est = np.zeros((len(measurements), 1))
        for i in range(len(measurements)):
            kf.predict()
            kf.update(measurements[i])
            est[i] = kf.x[0, 0]
    elapsed_ms = (time.perf_counter() - start) / reps * 1000.0

    return elapsed_ms, rmse(est[:, 0], true_states[:, 0, 0])


def bench_pykalman(reps=5):
    try:
        from pykalman import KalmanFilter as PKKalmanFilter
    except ImportError:
        return None

    dt, true_states, measurements, F, Q, H, R, x0, P0 = _task()
    z = measurements[:, 0, 0]

    start = time.perf_counter()
    for _ in range(reps):
        kf = PKKalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=x0[:, 0],
            initial_state_covariance=P0,
        )
        est_states, _ = kf.filter(z)
    elapsed_ms = (time.perf_counter() - start) / reps * 1000.0

    return elapsed_ms, rmse(est_states[:, 0], true_states[:, 0, 0])


def bench_simdkalman(reps=20, n_series=1):
    try:
        import simdkalman
    except ImportError:
        return None

    dt, true_states, measurements, F, Q, H, R, x0, P0 = _task()
    z = measurements[:, 0, 0].reshape(1, -1)

    kf = simdkalman.KalmanFilter(
        state_transition=F, process_noise=Q, observation_model=H, observation_noise=R
    )

    start = time.perf_counter()
    for _ in range(reps):
        result = kf.smooth(z, initial_value=x0[:, 0], initial_covariance=P0)
    elapsed_ms = (time.perf_counter() - start) / reps * 1000.0

    est = result.states.mean[0, :, 0]
    return elapsed_ms, rmse(est, true_states[:, 0, 0])


def bench_kalbee_vectorized(n_series=1000, reps=5):
    """Fair comparison to simdkalman's pitch: kalbee's own VectorizedKalmanFilter."""
    dt, true_states, measurements, F, Q, H, R, x0, P0 = _task()
    T = len(measurements)
    z = np.tile(measurements[:, :, 0], (n_series, 1, 1))  # (n_series, T, 1)

    start = time.perf_counter()
    for _ in range(reps):
        state = np.tile(x0, (n_series, 1, 1))
        cov = np.tile(P0, (n_series, 1, 1))
        vkf = VectorizedKalmanFilter(state, cov, F, Q, H, R)
        for t in range(T):
            vkf.predict()
            vkf.update(z[:, t, :].reshape(n_series, 1, 1))
    elapsed_ms = (time.perf_counter() - start) / reps * 1000.0

    return elapsed_ms


def bench_simdkalman_vectorized(n_series=1000, reps=5):
    """simdkalman's actual selling point: many independent series at once."""
    try:
        import simdkalman
    except ImportError:
        return None

    dt, true_states, measurements, F, Q, H, R, x0, P0 = _task()
    z = np.tile(measurements[:, 0, 0], (n_series, 1))

    kf = simdkalman.KalmanFilter(
        state_transition=F, process_noise=Q, observation_model=H, observation_noise=R
    )

    start = time.perf_counter()
    for _ in range(reps):
        kf.smooth(z, initial_value=x0[:, 0], initial_covariance=P0)
    elapsed_ms = (time.perf_counter() - start) / reps * 1000.0

    return elapsed_ms


def main():
    print("Single-series constant-velocity tracking (noisy sine, 200 steps, dt=0.05)")
    print(f"{'Library':<15} {'Time/run (ms)':>15} {'Position RMSE':>15}")
    print("-" * 47)

    for name, fn in [
        ("kalbee", bench_kalbee),
        ("filterpy", bench_filterpy),
        ("pykalman", bench_pykalman),
        ("simdkalman", bench_simdkalman),
    ]:
        result = fn()
        if result is None:
            print(f"{name:<15} {'(not installed)':>15}")
        else:
            ms, err = result
            print(f"{name:<15} {ms:>15.3f} {err:>15.4f}")

    print()
    print("Vectorized over many independent series (simdkalman's actual pitch)")
    n_series = 1000
    kalbee_loop_ms, _ = bench_kalbee(reps=5)
    kalbee_vec_ms = bench_kalbee_vectorized(n_series=n_series, reps=5)
    simd_vec_ms = bench_simdkalman_vectorized(n_series=n_series, reps=5)
    print(
        f"  kalbee, {n_series}x naive sequential loop: {kalbee_loop_ms * n_series:>10.1f} ms"
    )
    print(f"  kalbee, VectorizedKalmanFilter batch:  {kalbee_vec_ms:>10.1f} ms")
    if simd_vec_ms is not None:
        print(f"  simdkalman, {n_series} series at once:    {simd_vec_ms:>10.1f} ms")


if __name__ == "__main__":
    main()
