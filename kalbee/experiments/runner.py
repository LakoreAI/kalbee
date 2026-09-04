"""
Experiment runner for testing filters against synthetic signals.

Provides a simple API to run one or more filters against a generated signal
and compare their tracking performance.

Usage::

    from kalbee.experiments import run_experiment

    report = run_experiment(
        signal="sine",
        filters=["kf", "ekf", "ukf", "pf"],
        noise_std=0.5,
        duration=10.0,
    )
    print(report.summary())
"""

from typing import Dict, Optional, Sequence
import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.ekf_filter import ExtendedKalmanFilter
from kalbee.modules.filters.ukf_filter import UnscentedKalmanFilter
from kalbee.modules.filters.particle_filter import ParticleFilter
from kalbee.modules.filters.enkf_filter import EnsembleKalmanFilter
from kalbee.modules.filters.information_filter import InformationFilter
from kalbee.modules.filters.adaptive_kf import AdaptiveKalmanFilter
from kalbee.modules.filters.square_root_kf import SquareRootKalmanFilter

from kalbee.experiments.signals import SIGNALS
from kalbee.experiments.results import ExperimentResult, ExperimentReport


# Default constant-velocity model matrices
def _default_F(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix."""
    return np.array([[1.0, dt], [0.0, 1.0]])


def _default_Q(dt: float, q: float = 0.1) -> np.ndarray:
    """Discrete white noise process covariance for constant-velocity model."""
    return (
        np.array(
            [
                [dt**4 / 4, dt**3 / 2],
                [dt**3 / 2, dt**2],
            ]
        )
        * q
    )


def _default_H() -> np.ndarray:
    """Measure position only."""
    return np.array([[1.0, 0.0]])


# --- Non-linear function wrappers for EKF/UKF/PF ---


def _transition_func(x: np.ndarray, dt: float) -> np.ndarray:
    """Constant-velocity transition: x_new = F @ x."""
    F = _default_F(dt)
    return F @ x


def _transition_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """Jacobian of the transition function (linear, so it's just F)."""
    return _default_F(dt)


def _measurement_func(x: np.ndarray) -> np.ndarray:
    """Measure position: z = H @ x."""
    H = _default_H()
    return H @ x


def _measurement_jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian of measurement function (linear, so it's just H)."""
    return _default_H()


def _build_filter(
    name: str,
    dt: float,
    noise_std: float,
    process_noise: float,
    seed: Optional[int] = None,
) -> object:
    """
    Build a filter instance with default constant-velocity model.

    Args:
        name: Filter identifier (kf, ekf, ukf, pf, enkf, if, akf, srkf).
        dt: Time step.
        noise_std: Measurement noise standard deviation.
        process_noise: Process noise intensity.
        seed: Seed for the sampling-based filters (pf, enkf) for reproducibility.

    Returns:
        A filter instance ready for predict/update.
    """
    state = np.zeros((2, 1))
    cov = np.eye(2) * 10.0  # High initial uncertainty
    F = _default_F(dt)
    Q = _default_Q(dt, process_noise)
    H = _default_H()
    R = np.array([[noise_std**2]])

    name_clean = name.lower().replace("_", "").replace("-", "").replace(" ", "")

    if name_clean in ("kf", "kalmanfilter", "kalman"):
        return KalmanFilter(state, cov, F, Q, H, R)

    elif name_clean in ("ekf", "extendedkalmanfilter"):
        return ExtendedKalmanFilter(
            state,
            cov,
            Q,
            R,
            transition_function=_transition_func,
            measurement_function=_measurement_func,
            transition_jacobian=_transition_jacobian,
            measurement_jacobian=_measurement_jacobian,
        )

    elif name_clean in ("ukf", "unscentedkalmanfilter"):
        return UnscentedKalmanFilter(
            state,
            cov,
            Q,
            R,
            transition_function=_transition_func,
            measurement_function=_measurement_func,
        )

    elif name_clean in ("pf", "particlefilter", "particle"):
        return ParticleFilter(
            state=state,
            covariance=cov,
            transition_function=_transition_func,
            measurement_function=_measurement_func,
            measurement_covariance=R,
            num_particles=500,
            rng=seed,
            vectorized_functions=True,
        )

    elif name_clean in ("enkf", "ensemblekalmanfilter", "ensemble"):
        return EnsembleKalmanFilter(
            state=state,
            covariance=cov,
            transition_covariance=Q,
            measurement_covariance=R,
            transition_function=_transition_func,
            measurement_function=_measurement_func,
            ensemble_size=100,
            rng=seed,
            vectorized_functions=True,
        )

    elif name_clean in ("if", "informationfilter", "information"):
        return InformationFilter(state, cov, F, Q, H, R)

    elif name_clean in ("akf", "adaptivekalmanfilter", "adaptive"):
        return AdaptiveKalmanFilter(state, cov, F, Q, H, R, window_size=10)

    elif name_clean in ("srkf", "squarerootkalmanfilter", "squareroot"):
        return SquareRootKalmanFilter(state, cov, F, Q, H, R)

    else:
        raise ValueError(
            f"Unknown filter: '{name}'. Options: kf, ekf, ukf, pf, enkf, if, akf, srkf."
        )


def _run_single_filter(
    filter_obj,
    filter_name: str,
    time_steps: np.ndarray,
    true_states: np.ndarray,
    measurements: np.ndarray,
    dt: float,
) -> ExperimentResult:
    """Run a single filter over the signal and collect results."""
    T = len(time_steps)
    n = 2  # state dimension

    estimated_states = np.zeros((T, n, 1))
    covariances = []

    for k in range(T):
        # Predict
        filter_obj.predict(dt=dt)

        # Update with measurement
        z = measurements[k]
        filter_obj.update(z)

        # Store results
        estimated_states[k] = filter_obj.state.reshape(n, 1)
        covariances.append(filter_obj.covariance.copy())

    return ExperimentResult(
        filter_name=filter_name,
        time_steps=time_steps,
        true_states=true_states,
        estimated_states=estimated_states,
        measurements=measurements,
        covariances=covariances,
    )


def run_experiment(
    signal: str = "sine",
    filters: Optional[Sequence[str]] = None,
    duration: float = 10.0,
    dt: float = 0.1,
    noise_std: float = 0.3,
    process_noise: float = 0.1,
    seed: Optional[int] = 42,
    signal_kwargs: Optional[Dict] = None,
) -> ExperimentReport:
    """
    Run a tracking experiment comparing multiple filters on a synthetic signal.

    Args:
        signal: Signal type — "sine", "cosine", "linear", "step", "maneuver".
        filters: List of filter names to compare.
                 Default: ["kf", "ekf", "ukf"].
                 Options: "kf", "ekf", "ukf", "pf", "enkf", "if", "akf", "srkf".
        duration: Signal duration in seconds.
        dt: Time step.
        noise_std: Measurement noise standard deviation.
        process_noise: Process noise intensity.
        seed: Random seed for reproducibility.
        signal_kwargs: Extra kwargs passed to the signal generator.

    Returns:
        ExperimentReport with results for each filter.

    Example::

        from kalbee.experiments import run_experiment

        report = run_experiment(
            signal="sine",
            filters=["kf", "ekf", "ukf", "pf"],
            noise_std=0.5,
        )
        print(report.summary())
    """
    if filters is None:
        filters = ["kf", "ekf", "ukf"]

    # Generate signal
    sig_kwargs = dict(duration=duration, dt=dt, noise_std=noise_std, seed=seed)
    if signal_kwargs:
        sig_kwargs.update(signal_kwargs)

    if signal in SIGNALS:
        sig_func = SIGNALS[signal]
    else:
        raise ValueError(
            f"Unknown signal: '{signal}'. Options: {list(SIGNALS.keys())}. "
            "Use custom_signal() for custom functions."
        )

    time_steps, true_states, measurements = sig_func(**sig_kwargs)

    # Run each filter
    results = []
    for filter_name in filters:
        filt = _build_filter(filter_name, dt, noise_std, process_noise, seed=seed)
        result = _run_single_filter(
            filt,
            filter_name.upper(),
            time_steps,
            true_states,
            measurements,
            dt,
        )
        results.append(result)

    # Signal params for the report
    signal_params = {
        "duration": duration,
        "dt": dt,
        "noise_std": noise_std,
        "process_noise": process_noise,
    }
    if signal_kwargs:
        signal_params.update(signal_kwargs)

    return ExperimentReport(
        signal_name=signal,
        results=results,
        signal_params=signal_params,
    )
