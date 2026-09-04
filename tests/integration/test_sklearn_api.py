import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from kalbee.modules.integration.sklearn_api import KalmanEstimator  # noqa: E402
from kalbee.modules.utils.metrics import rmse  # noqa: E402


def _noisy_sine(n=200, noise_std=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n)
    true = np.sin(t)
    noisy = true + rng.standard_normal(n) * noise_std
    return t, true, noisy


def test_fit_transform_reduces_noise_1d():
    t, true, noisy = _noisy_sine()
    dt = t[1] - t[0]
    est = KalmanEstimator(dt=dt, process_var=5.0, measurement_var=0.2)
    smoothed = est.fit_transform(noisy)

    assert smoothed.shape == (len(noisy), 1)
    assert rmse(smoothed[:, 0], true) < rmse(noisy, true)


def test_predict_is_alias_for_transform():
    _, _, noisy = _noisy_sine()
    est = KalmanEstimator().fit(noisy)
    assert np.allclose(est.transform(noisy), est.predict(noisy))


def test_multi_dim_input():
    rng = np.random.default_rng(1)
    T = 50
    true = np.stack([np.linspace(0, 10, T), np.linspace(5, -5, T)], axis=1)
    noisy = true + rng.standard_normal(true.shape) * 0.3

    est = KalmanEstimator(process_var=0.05, measurement_var=0.3)
    smoothed = est.fit_transform(noisy)

    assert smoothed.shape == (T, 2)


def test_return_full_state():
    _, _, noisy = _noisy_sine(n=30)
    est = KalmanEstimator(return_full_state=True).fit(noisy)
    full = est.transform(noisy)
    assert full.shape == (30, 2)  # [position, velocity]


def test_get_set_params_sklearn_compatible():
    est = KalmanEstimator(mode="akf", process_var=2.0)
    params = est.get_params()
    assert params["mode"] == "akf"
    assert params["process_var"] == 2.0

    est.set_params(process_var=5.0)
    assert est.process_var == 5.0


def test_works_inside_sklearn_pipeline():
    from sklearn.pipeline import Pipeline

    _, _, noisy = _noisy_sine(n=60)
    pipe = Pipeline(
        [("kalman", KalmanEstimator(process_var=0.05, measurement_var=0.2))]
    )
    out = pipe.fit_transform(noisy)
    assert out.shape == (60, 1)


def test_transform_before_fit_raises():
    est = KalmanEstimator()
    with pytest.raises(Exception):
        est.transform(np.array([1.0, 2.0, 3.0]))


def test_tune_true_uses_quick_tune():
    _, _, noisy = _noisy_sine(n=100)
    est = KalmanEstimator(tune=True).fit(noisy)
    smoothed = est.transform(noisy)
    assert smoothed.shape == (100, 1)


def test_transform_rejects_mismatched_feature_count():
    _, _, noisy = _noisy_sine(n=30)
    est = KalmanEstimator().fit(noisy)  # fit on 1 feature
    with pytest.raises(ValueError, match="features"):
        est.transform(np.zeros((30, 2)))  # transform with 2 features
