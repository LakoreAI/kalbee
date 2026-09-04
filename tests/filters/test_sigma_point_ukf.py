"""Tests for the SigmaPointUKF (UKF with pluggable sigma-point strategies)."""

import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter  # noqa: F401
from kalbee.modules.filters.sigma_point_ukf import SigmaPointUKF
from kalbee.modules.filters.sigma_points import (
    SimplexSigmaPoints,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)


class TestSigmaPointUKF:
    def test_initialization_default_sigma(self):
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
        )
        assert isinstance(ukf.sigma_points, SimplexSigmaPoints)

    def test_initialization_custom_sigma(self):
        sp = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0.0)
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            sigma_points=sp,
        )
        assert isinstance(ukf.sigma_points, MerweScaledSigmaPoints)

    def test_predict_update_cycle(self):
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2) * 10.0,
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: np.array(
                [[x[0, 0] + x[1, 0] * dt], [x[1, 0]]]
            ),
            measurement_function=lambda x: x[:1],
        )
        for _ in range(10):
            ukf.predict()
            ukf.update(np.array([[1.0]]))
        assert np.abs(ukf.state[0, 0] - 1.0) < 2.0

    def test_julier_sigma_points(self):
        sp = JulierSigmaPoints(n=2, kappa=0.0)
        ukf = SigmaPointUKF(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            sigma_points=sp,
        )
        ukf.predict()
        assert ukf.state.shape == (2, 1)
