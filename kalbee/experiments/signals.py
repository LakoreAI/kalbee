"""
Signal generators for filter experiments.

Provides configurable signal functions that return ground-truth states
and noisy measurements for testing filter tracking performance.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Union

# Accepts an int seed, an existing Generator, or None (fresh generator).
RngLike = Optional[Union[int, np.random.Generator]]


def sine_signal(
    duration: float = 10.0,
    dt: float = 0.1,
    amplitude: float = 1.0,
    frequency: float = 0.5,
    phase: float = 0.0,
    noise_std: float = 0.3,
    seed: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a sine wave signal with noisy measurements.

    Args:
        duration: Total signal duration in seconds.
        dt: Time step between samples.
        amplitude: Amplitude of the sine wave.
        frequency: Frequency in Hz.
        phase: Phase offset in radians.
        noise_std: Standard deviation of measurement noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time_steps, true_states, measurements).
        - time_steps: shape (T,)
        - true_states: shape (T, 2, 1) — [position, velocity]
        - measurements: shape (T, 1, 1) — noisy position
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, dt)
    T = len(t)
    omega = 2 * np.pi * frequency

    # True position and velocity
    position = amplitude * np.sin(omega * t + phase)
    velocity = amplitude * omega * np.cos(omega * t + phase)

    true_states = np.zeros((T, 2, 1))
    true_states[:, 0, 0] = position
    true_states[:, 1, 0] = velocity

    measurements = np.zeros((T, 1, 1))
    measurements[:, 0, 0] = position + rng.standard_normal(T) * noise_std

    return t, true_states, measurements


def cosine_signal(
    duration: float = 10.0,
    dt: float = 0.1,
    amplitude: float = 1.0,
    frequency: float = 0.5,
    phase: float = 0.0,
    noise_std: float = 0.3,
    seed: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a cosine wave signal with noisy measurements.

    Returns same format as sine_signal.
    """
    return sine_signal(
        duration=duration,
        dt=dt,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase + np.pi / 2,
        noise_std=noise_std,
        seed=seed,
    )


def linear_signal(
    duration: float = 10.0,
    dt: float = 0.1,
    slope: float = 1.0,
    intercept: float = 0.0,
    noise_std: float = 0.3,
    seed: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a constant-velocity linear signal with noisy measurements.

    Returns:
        Tuple of (time_steps, true_states, measurements).
        - true_states: shape (T, 2, 1) — [position, velocity]
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, dt)
    T = len(t)

    position = intercept + slope * t
    velocity = np.full(T, slope)

    true_states = np.zeros((T, 2, 1))
    true_states[:, 0, 0] = position
    true_states[:, 1, 0] = velocity

    measurements = np.zeros((T, 1, 1))
    measurements[:, 0, 0] = position + rng.standard_normal(T) * noise_std

    return t, true_states, measurements


def step_signal(
    duration: float = 10.0,
    dt: float = 0.1,
    step_time: float = 5.0,
    low_value: float = 0.0,
    high_value: float = 1.0,
    noise_std: float = 0.3,
    seed: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a step signal (sudden jump) with noisy measurements.

    Returns:
        Tuple of (time_steps, true_states, measurements).
        - true_states: shape (T, 2, 1) — [position, velocity=0]
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, dt)
    T = len(t)

    position = np.where(t >= step_time, high_value, low_value)

    true_states = np.zeros((T, 2, 1))
    true_states[:, 0, 0] = position

    measurements = np.zeros((T, 1, 1))
    measurements[:, 0, 0] = position + rng.standard_normal(T) * noise_std

    return t, true_states, measurements


def custom_signal(
    func: Callable[[np.ndarray], np.ndarray],
    derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    duration: float = 10.0,
    dt: float = 0.1,
    noise_std: float = 0.3,
    seed: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a custom signal from a user-defined function.

    Args:
        func: f(t) -> position values (accepts array of time steps).
        derivative: f'(t) -> velocity values. If None, estimated numerically.
        duration: Total signal duration.
        dt: Time step.
        noise_std: Measurement noise std.
        seed: Random seed.

    Returns:
        Tuple of (time_steps, true_states, measurements).
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, dt)
    T = len(t)

    position = func(t)

    if derivative is not None:
        velocity = derivative(t)
    else:
        # Numerical differentiation
        velocity = np.gradient(position, dt)

    true_states = np.zeros((T, 2, 1))
    true_states[:, 0, 0] = position
    true_states[:, 1, 0] = velocity

    measurements = np.zeros((T, 1, 1))
    measurements[:, 0, 0] = position + rng.standard_normal(T) * noise_std

    return t, true_states, measurements


def maneuver_signal(
    duration: float = 15.0,
    dt: float = 0.1,
    maneuver_start: float = 5.0,
    maneuver_end: float = 10.0,
    initial_velocity: float = 2.0,
    acceleration: float = 1.5,
    noise_std: float = 0.5,
    seed: RngLike = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a maneuvering target signal (CV -> CA -> CV) with noisy measurements.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, dt)
    T = len(t)

    true_states = np.zeros((T, 2, 1))
    x = 0.0
    v = initial_velocity

    for i, step in enumerate(t):
        if maneuver_start <= step < maneuver_end:
            a_step = acceleration
        else:
            a_step = 0.0

        v += a_step * dt
        x += v * dt + 0.5 * a_step * dt**2

        true_states[i, 0, 0] = x
        true_states[i, 1, 0] = v

    measurements = np.zeros((T, 1, 1))
    measurements[:, 0, 0] = true_states[:, 0, 0] + rng.standard_normal(T) * noise_std

    return t, true_states, measurements


# Convenience dictionary for string-based access
SIGNALS = {
    "sine": sine_signal,
    "cosine": cosine_signal,
    "linear": linear_signal,
    "step": step_signal,
    "maneuver": maneuver_signal,
}
