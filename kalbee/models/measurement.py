"""
Ready-made measurement models and process-noise builders.

These helpers produce the ``H``/``R`` matrices (and the process-noise ``Q``)
that filter users would otherwise hand-assemble. State vectors follow the
per-axis block convention used by :mod:`kalbee.models.motion`: for ``n_dims``
spatial axes and a kinematic ``order`` (1 = [pos, vel], 2 = [pos, vel, acc]),
the state is::

    [p_1, v_1, (a_1), p_2, v_2, (a_2), ...]

i.e. one contiguous kinematic block per axis.
"""

from typing import Tuple
import numpy as np
from scipy.linalg import block_diag


def discrete_white_noise(order: int, dt: float, var: float = 1.0) -> np.ndarray:
    """
    Build the single-axis discrete white-noise process covariance ``Q``.

    Args:
        order: Kinematic order — 1 for constant-velocity (2x2), 2 for
               constant-acceleration (3x3).
        dt: Time step.
        var: Process-noise variance (spectral intensity).

    Returns:
        A single-axis ``Q`` block of shape (order+1, order+1).
    """
    if order == 1:
        q = np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0],
                [dt**3 / 2.0, dt**2],
            ]
        )
    elif order == 2:
        q = np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0, dt**2 / 2.0],
                [dt**3 / 2.0, dt**2, dt],
                [dt**2 / 2.0, dt, 1.0],
            ]
        )
    else:
        raise ValueError(f"order must be 1 or 2, got {order}.")
    return q * var


def _block_per_axis(single_axis: np.ndarray, n_dims: int) -> np.ndarray:
    """Replicate a single-axis block into a block-diagonal multi-axis matrix."""
    if n_dims == 1:
        return single_axis
    return block_diag(*([single_axis] * n_dims))


def position_measurement_model(
    order: int = 1,
    n_dims: int = 1,
    measurement_var: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measurement model that observes position on every axis.

    Args:
        order: Kinematic order of the state (1 = CV, 2 = CA).
        n_dims: Number of spatial axes.
        measurement_var: Measurement-noise variance (per axis).

    Returns:
        Tuple ``(H, R)`` where ``H`` has shape (n_dims, n_dims * (order + 1))
        and ``R`` has shape (n_dims, n_dims).
    """
    block_size = order + 1
    # Single-axis row selecting position (the first element of the block).
    row = np.zeros((1, block_size))
    row[0, 0] = 1.0
    H = _block_per_axis(row, n_dims)
    R = np.eye(n_dims) * measurement_var
    return H, R
