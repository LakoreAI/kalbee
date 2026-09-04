"""
Multi-object tracking demo.

Simulates several targets crossing a 2-D scene, feeds noisy per-frame position
detections (in shuffled order, as a real detector would emit them) to
``MultiObjectTracker``, and reports how well identities are maintained.

Run with:
    python examples/multi_object_tracking.py
"""

import numpy as np

from kalbee import KalmanFilter, MultiObjectTracker
from kalbee.models import constant_velocity, position_measurement_model


def build_tracker(dt: float) -> MultiObjectTracker:
    """A SORT-style tracker over a 2-D constant-velocity Kalman filter."""
    F, Q = constant_velocity(dt=dt, process_var=0.05, n_dims=2)
    H, R = position_measurement_model(order=1, n_dims=2, measurement_var=0.5)

    def new_track(z):
        # Seed state [x, vx, y, vy] at the detection with zero initial velocity.
        x0 = np.array([[z[0]], [0.0], [z[1]], [0.0]])
        return KalmanFilter(x0, np.eye(4) * 10.0, F, Q, H, R)

    return MultiObjectTracker(new_track, n_init=3, max_age=5, gate=6.0)


def main() -> None:
    rng = np.random.default_rng(42)
    dt = 1.0
    n_frames = 30
    noise_std = 0.3

    # Three targets: position (x, y) and constant velocity (vx, vy).
    targets = [
        {"p": np.array([0.0, 0.0]), "v": np.array([1.0, 0.5])},
        {"p": np.array([30.0, 0.0]), "v": np.array([-1.0, 0.5])},  # crosses target 0
        {"p": np.array([0.0, 20.0]), "v": np.array([1.2, -0.3])},
    ]

    tracker = build_tracker(dt)

    print(f"Tracking {len(targets)} targets over {n_frames} frames\n")
    print(f"{'frame':>5} | {'#dets':>5} | {'#confirmed':>10}")
    print("-" * 28)

    for frame in range(n_frames):
        # Advance ground truth.
        for tgt in targets:
            tgt["p"] = tgt["p"] + tgt["v"] * dt

        # Emit noisy detections in randomized order (detectors have no track order).
        dets = np.array([t["p"] + rng.standard_normal(2) * noise_std for t in targets])
        rng.shuffle(dets)

        confirmed = tracker.update(dets, dt=dt)

        if frame % 5 == 0 or frame == n_frames - 1:
            print(f"{frame:>5} | {len(dets):>5} | {len(confirmed):>10}")

    # Final report: match each confirmed track to its nearest true target.
    print("\nFinal confirmed tracks:")
    truth = np.array([t["p"] for t in targets])
    for t in tracker.confirmed_tracks():
        est = np.array([t.state[0, 0], t.state[2, 0]])
        d = np.linalg.norm(truth - est, axis=1)
        nearest = int(np.argmin(d))
        print(
            f"  track id={t.id}  est=({est[0]:6.2f}, {est[1]:6.2f})  "
            f"nearest truth #{nearest}=({truth[nearest][0]:6.2f}, "
            f"{truth[nearest][1]:6.2f})  err={d[nearest]:.3f}  hits={t.hits}"
        )


if __name__ == "__main__":
    main()
