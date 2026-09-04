"""
Generate the animated ``.gif`` demos used across the documentation.

Each figure is built with kalbee itself, then saved as an animated GIF with
``matplotlib.animation.PillowWriter`` so the docs can show the filters
*working* instead of only a final screenshot::

    uv run python scripts/generate_demo_gif.py

Outputs (into ``docs/assets/gif``):

- ``filter_demo.gif``     a Kalman filter denoising a noisy 1-D signal
- ``imm_maneuver.gif``    KF vs. IMM on a 2-D target that executes a turn

The real-video multi-object-tracking demo (``mot16_tracking.gif``) is built
separately by ``scripts/mot16_demo.py`` from a public MOT Challenge sequence.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from kalbee import KalmanFilter, InteractingMultipleModel
from kalbee.models import constant_velocity, position_measurement_model

GIF_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "gif")


def _save(fig, anim, name, fps=12, dpi=80):
    os.makedirs(GIF_DIR, exist_ok=True)
    path = os.path.join(GIF_DIR, name)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"wrote {os.path.abspath(path)}")


def filter_demo():
    """A 1-D constant-velocity KF smoothing a noisy sine-like signal."""
    dt = 0.05
    t = np.arange(0, 8.0, dt)
    truth = np.sin(t * 1.2) + 0.25 * t

    rng = np.random.default_rng(42)
    z = truth + rng.standard_normal(len(t)) * 0.35

    F, Q = constant_velocity(dt=dt, process_var=0.02, n_dims=1)
    H, R = position_measurement_model(order=1, n_dims=1, measurement_var=0.35**2)
    kf = KalmanFilter(np.array([[0.0], [0.6]]), np.eye(2) * 5.0, F, Q, H, R)

    est, var, vel = [], [], []
    for zk in z:
        kf.predict()
        kf.update(np.array([[zk]]))
        est.append(kf.x[0, 0])
        var.append(kf.P[0, 0])
        vel.append(kf.x[1, 0])
    est = np.asarray(est)
    var = np.asarray(var)
    vel = np.asarray(vel)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 4.4), sharex=True, constrained_layout=True
    )
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(truth.min() - 1.0, truth.max() + 1.0)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(-2.0, 2.4)

    def anim(i):
        k = i + 1
        ax1.clear()
        ax2.clear()
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(truth.min() - 1.0, truth.max() + 1.0)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_ylim(-2.0, 2.4)

        (h_meas,) = ax1.plot(t[:k], z[:k], ".", ms=4, color="#b8860b", alpha=0.8)
        (h_true,) = ax1.plot(t[:k], truth[:k], color="0.35", lw=1.2)
        ax1.fill_between(
            t[:k],
            (est - var**0.5)[:k],
            (est + var**0.5)[:k],
            color="#1565c0",
            alpha=0.12,
        )
        (h_filt,) = ax1.plot(t[:k], est[:k], color="#1565c0", lw=1.8)
        h_band = ax1.collections[-1]
        ax1.legend(
            [h_filt, h_meas, h_true, h_band],
            ["filtered", "measurements", "true", "±1σ"],
            loc="upper right",
            fontsize=7,
            ncol=4,
            frameon=False,
        )

        (h_v,) = ax2.plot(t[:k], vel[:k], color="#2e7d32", lw=1.6)
        (h_vt,) = ax2.plot(
            t[:k], np.gradient(truth, dt)[:k], color="0.35", lw=1.0, ls=":"
        )
        ax2.legend(
            [h_v, h_vt],
            ["velocity (est.)", "velocity (true)"],
            loc="upper right",
            fontsize=7,
            ncol=2,
            frameon=False,
        )
        ax1.set_ylabel("position")
        ax2.set_ylabel("velocity")
        ax2.set_xlabel("time (s)")

    return _save(
        fig, FuncAnimation(fig, anim, frames=len(t), blit=False), "filter_demo.gif"
    )


def imm_maneuver():
    """KF (constant-velocity) vs IMM (CV + CA) on a target that turns sharply."""
    dt = 0.1
    n = 220
    rng = np.random.default_rng(7)

    pos = np.array([0.0, 0.0])
    vel = np.array([2.0, 0.4])
    truth, zs = [], []
    for k in range(n):
        if 100 <= k < 160:  # lateral acceleration (a turn)
            vel += np.array([-0.35, 2.4]) * dt
        pos = pos + vel * dt
        truth.append(pos.copy())
        zs.append(pos + rng.standard_normal(2) * 0.3)
    truth = np.array(truth)
    zs = np.array(zs)

    # Baseline KF: constant velocity, state [x, vx, y, vy]
    F4, Q4 = constant_velocity(dt=dt, process_var=0.15, n_dims=2)
    H2, R2 = position_measurement_model(order=1, n_dims=2, measurement_var=0.09)
    kf = KalmanFilter(
        np.array([[0.0], [2.0], [0.0], [0.4]]), np.eye(4) * 4.0, F4, Q4, H2, R2
    )

    # IMM: CV + CA on a 6-D state [x, vx, y, vy, ax, ay]
    F6 = np.array(
        [
            [1, 0, dt, 0, 0.5 * dt**2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    H6 = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

    def _kf6(q):
        return KalmanFilter(
            np.array([[0.0], [2.0], [0.0], [0.4], [0.0], [0.0]]),
            np.eye(6) * 4.0,
            F6,
            np.eye(6) * q,
            H6,
            R2,
        )

    imm = InteractingMultipleModel(
        [_kf6(0.02), _kf6(2.0)],
        np.array([[0.97, 0.03], [0.03, 0.97]]),
        np.array([0.8, 0.2]),
    )

    kf_track, imm_track = [], []
    for zk in zs:
        kf.predict()
        kf.update(zk.reshape(-1, 1))
        imm.predict()
        imm.update(zk.reshape(-1, 1))
        kf_track.append(kf.x[[0, 2], 0].copy())
        imm_track.append(imm.x[[0, 2], 0].copy())
    kf_track = np.array(kf_track)
    imm_track = np.array(imm_track)

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 5.0), constrained_layout=True)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1.5, 11.0)

    def anim(i):
        k = i + 1
        ax.clear()
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-1.5, 11.0)

        (h_true,) = ax.plot(truth[:k, 0], truth[:k, 1], lw=1.0, color="0.35")
        (h_meas,) = ax.plot(zs[:k, 0], zs[:k, 1], ".", ms=3, color="#b8860b", alpha=0.7)
        (h_kf,) = ax.plot(kf_track[:k, 0], kf_track[:k, 1], color="#c62828", lw=1.6)
        (h_imm,) = ax.plot(imm_track[:k, 0], imm_track[:k, 1], color="#1565c0", lw=1.8)
        ax.legend(
            [h_imm, h_kf, h_meas, h_true],
            [
                "IMM (CV+CA) — adapts to the turn",
                "KF (CV) — lags",
                "measurements",
                "true",
            ],
            loc="upper left",
            fontsize=7,
            frameon=False,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            "A single KF model cannot follow a maneuver; the IMM blends two models",
            fontsize=8,
        )

    return _save(
        fig,
        FuncAnimation(fig, anim, frames=len(truth), blit=False),
        "imm_maneuver.gif",
    )


if __name__ == "__main__":
    os.makedirs(GIF_DIR, exist_ok=True)
    filter_demo()
    imm_maneuver()
