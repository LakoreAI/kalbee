"""Tests for the pluggable sigma-point strategies."""

import numpy as np
import pytest

from kalbee.modules.filters.sigma_points import (
    SimplexSigmaPoints,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
)


class TestSimplexSigmaPoints:
    def test_sigma_points_shape(self):
        sp = SimplexSigmaPoints(n=2, alpha=0.001, beta=2.0, kappa=0.0)
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        sigmas = sp.sigma_points(x, P)
        assert sigmas.shape == (5, 2)

    def test_sigma_points_mean(self):
        sp = SimplexSigmaPoints(n=2, alpha=0.001, beta=2.0, kappa=0.0)
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        sigmas = sp.sigma_points(x, P)
        # Weighted mean should recover the original mean
        mean = np.dot(sp.weights_mean, sigmas)
        assert np.allclose(mean, x, atol=1e-10)

    def test_num_sigma_points(self):
        sp = SimplexSigmaPoints(n=3)
        assert sp.num_sigma_points == 7


class TestMerweScaledSigmaPoints:
    def test_sigma_points_shape(self):
        sp = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=0.0)
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3)
        sigmas = sp.sigma_points(x, P)
        assert sigmas.shape == (7, 3)

    def test_weights_sum_to_one(self):
        sp = MerweScaledSigmaPoints(n=2)
        assert np.sum(sp.weights_mean) == pytest.approx(1.0)


class TestJulierSigmaPoints:
    def test_sigma_points_shape(self):
        sp = JulierSigmaPoints(n=2, kappa=0.0)
        x = np.array([1.0, 2.0])
        P = np.eye(2)
        sigmas = sp.sigma_points(x, P)
        assert sigmas.shape == (5, 2)

    def test_weights_symmetry(self):
        sp = JulierSigmaPoints(n=2, kappa=0.0)
        # Julier weights should be symmetric (except for the first)
        assert sp.weights_mean[1] == pytest.approx(sp.weights_mean[-1])
