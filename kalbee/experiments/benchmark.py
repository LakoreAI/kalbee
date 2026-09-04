import time
from typing import Dict

from kalbee.experiments.runner import run_experiment


def run_benchmark(
    duration: float = 10.0,
    dt: float = 0.05,
    noise_std: float = 0.3,
    process_noise: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """
    Run tracking benchmarks comparing speed and accuracy of all filters.

    Returns:
        Dictionary mapping filter names to benchmark metrics.
    """
    filters = ["kf", "ekf", "ukf", "pf", "enkf", "if", "akf", "srkf"]
    results = {}

    print(f"Running benchmarks over {int(duration / dt)} steps...")
    print(f"{'Filter':<25} | {'Execution Time (ms)':<20} | {'Position RMSE':<15}")
    print("-" * 70)

    for name in filters:
        start_time = time.perf_counter()

        # Run experiment (which builds the filter and steps it over the signal)
        report = run_experiment(
            signal="sine",
            filters=[name],
            duration=duration,
            dt=dt,
            noise_std=noise_std,
            process_noise=process_noise,
            seed=42,
        )

        elapsed = (time.perf_counter() - start_time) * 1000.0  # ms
        res = report.results[0]
        pos_rmse = res.position_rmse()

        results[name.upper()] = {
            "time_ms": elapsed,
            "rmse": pos_rmse,
        }

        print(f"{name.upper():<25} | {elapsed:<20.2f} | {pos_rmse:<15.4f}")

    return results


if __name__ == "__main__":
    run_benchmark()
