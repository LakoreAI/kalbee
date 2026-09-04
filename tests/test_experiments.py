import numpy as np
import pytest
from kalbee.experiments import run_experiment
from kalbee.experiments.signals import (
    sine_signal,
    cosine_signal,
    linear_signal,
    step_signal,
    custom_signal,
)


# --- Signal Tests ---


def test_sine_signal_shape():
    t, states, measurements = sine_signal(duration=5.0, dt=0.1, seed=42)

    assert t.shape == (50,)
    assert states.shape == (50, 2, 1)
    assert measurements.shape == (50, 1, 1)


def test_cosine_signal():
    t, states, measurements = cosine_signal(duration=2.0, dt=0.1, seed=42)

    assert len(t) == 20
    # Cosine starts at amplitude
    assert np.isclose(states[0, 0, 0], 1.0, atol=0.1)


def test_linear_signal():
    t, states, measurements = linear_signal(
        duration=5.0, dt=1.0, slope=2.0, intercept=1.0, noise_std=0.0, seed=42
    )

    assert len(t) == 5
    # Position at t=0 should be intercept=1
    assert np.isclose(states[0, 0, 0], 1.0)
    # Velocity should be slope=2
    assert np.isclose(states[0, 1, 0], 2.0)


def test_step_signal():
    t, states, measurements = step_signal(
        duration=10.0,
        dt=1.0,
        step_time=5.0,
        low_value=0.0,
        high_value=10.0,
        noise_std=0.0,
        seed=42,
    )

    assert states[0, 0, 0] == 0.0  # Before step
    assert states[5, 0, 0] == 10.0  # At step time


def test_custom_signal():
    t, states, measurements = custom_signal(
        func=lambda t: np.sin(t),
        derivative=lambda t: np.cos(t),
        duration=3.0,
        dt=0.1,
        noise_std=0.0,
        seed=42,
    )

    assert states.shape[0] == 30
    assert np.isclose(states[0, 0, 0], 0.0, atol=0.01)


# --- Runner Tests ---


def test_run_experiment_default():
    report = run_experiment(
        signal="sine",
        filters=["kf", "ekf"],
        duration=5.0,
        dt=0.1,
        seed=42,
    )

    assert report.signal_name == "sine"
    assert len(report.results) == 2
    assert report.results[0].filter_name == "KF"
    assert report.results[1].filter_name == "EKF"


def test_run_experiment_all_filters():
    report = run_experiment(
        signal="sine",
        filters=["kf", "ekf", "ukf", "pf", "enkf", "if", "akf"],
        duration=3.0,
        dt=0.1,
        seed=42,
    )

    assert len(report.results) == 7

    # All filters should produce finite results
    for result in report.results:
        assert np.all(np.isfinite(result.estimated_states))
        assert result.position_rmse() < 10.0  # Reasonable bound


def test_run_experiment_summary():
    report = run_experiment(
        signal="cosine",
        filters=["kf", "ekf"],
        duration=3.0,
        seed=42,
    )

    summary = report.summary()
    assert "Experiment Report" in summary
    assert "cosine" in summary
    assert "KF" in summary
    assert "EKF" in summary
    assert "Best position tracking" in summary


def test_run_experiment_to_dict():
    report = run_experiment(
        signal="linear",
        filters=["kf"],
        duration=3.0,
        seed=42,
    )

    d = report.to_dict()
    assert d["signal"] == "linear"
    assert len(d["results"]) == 1
    assert "position_rmse" in d["results"][0]


def test_run_experiment_invalid_signal():
    with pytest.raises(ValueError, match="Unknown signal"):
        run_experiment(signal="invalid_signal")


def test_run_experiment_invalid_filter():
    with pytest.raises(ValueError, match="Unknown filter"):
        run_experiment(filters=["unknown_filter"])


def test_experiment_result_nees():
    report = run_experiment(
        signal="sine",
        filters=["kf"],
        duration=3.0,
        seed=42,
    )

    result = report.results[0]
    nees_vals = result.nees_values()

    assert len(nees_vals) == len(result.time_steps)
    assert result.average_nees() > 0


def test_get_best_filter():
    report = run_experiment(
        signal="sine",
        filters=["kf", "ekf", "ukf"],
        duration=5.0,
        seed=42,
    )

    best = report.get_best_filter()
    assert best.filter_name in ["KF", "EKF", "UKF"]
    assert best.position_rmse() == min(r.position_rmse() for r in report.results)
