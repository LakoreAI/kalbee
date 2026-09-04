"""Tests for AutoFilter covering all filter modes."""

import numpy as np
import pytest
from kalbee.modules.filters.auto_filter import AutoFilter
from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.ekf_filter import ExtendedKalmanFilter
from kalbee.modules.filters.ukf_filter import UnscentedKalmanFilter
from kalbee.modules.filters.abg_filter import AlphaBetaGammaFilter
from kalbee.modules.filters.particle_filter import ParticleFilter
from kalbee.modules.filters.enkf_filter import EnsembleKalmanFilter
from kalbee.modules.filters.information_filter import InformationFilter
from kalbee.modules.filters.adaptive_kf import AdaptiveKalmanFilter
from kalbee.modules.filters.square_root_kf import SquareRootKalmanFilter
from kalbee.modules.filters.vectorized_kf import VectorizedKalmanFilter
from kalbee.modules.filters.fading_memory_kf import FadingMemoryKalmanFilter
from kalbee.modules.filters.hinfinity_filter import HInfinityFilter
from kalbee.modules.filters.sigma_point_ukf import SigmaPointUKF


# --- Shared fixtures ---


@pytest.fixture
def linear_args():
    """Standard linear filter arguments."""
    state = np.array([[0.0], [1.0]])
    cov = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])
    return state, cov, F, Q, H, R


# --- KF modes ---


class TestAutoFilterKF:
    def test_kf_short(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args, mode="kf")
        assert isinstance(kf, KalmanFilter)

    def test_kf_full_name(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args, mode="KalmanFilter")
        assert isinstance(kf, KalmanFilter)

    def test_kf_alias(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args, mode="kalman")
        assert isinstance(kf, KalmanFilter)

    def test_kf_with_underscores(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args, mode="kalman_filter")
        assert isinstance(kf, KalmanFilter)

    def test_kf_with_dashes(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args, mode="kalman-filter")
        assert isinstance(kf, KalmanFilter)

    def test_kf_uppercase(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args, mode="KF")
        assert isinstance(kf, KalmanFilter)

    def test_default_mode(self, linear_args):
        kf = AutoFilter.from_filter(*linear_args)
        assert isinstance(kf, KalmanFilter)


# --- EKF modes ---


class TestAutoFilterEKF:
    def test_ekf_short(self):
        state = np.array([[0.0]])
        cov = np.eye(1)
        Q = np.eye(1) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x
        F = lambda x, dt: np.eye(1)
        H = lambda x: np.eye(1)
        ekf = AutoFilter.from_filter(state, cov, Q, R, f, h, F, H, mode="ekf")
        assert isinstance(ekf, ExtendedKalmanFilter)

    def test_ekf_full_name(self):
        state = np.array([[0.0]])
        cov = np.eye(1)
        Q = np.eye(1) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x
        F = lambda x, dt: np.eye(1)
        H = lambda x: np.eye(1)
        ekf = AutoFilter.from_filter(
            state, cov, Q, R, f, h, F, H, mode="ExtendedKalmanFilter"
        )
        assert isinstance(ekf, ExtendedKalmanFilter)


# --- ABG mode ---


class TestAutoFilterABG:
    def test_abg_short(self):
        abg = AutoFilter.from_filter(
            np.array([[0], [0], [0]]), 0.1, 0.1, 0.05, mode="abg"
        )
        assert isinstance(abg, AlphaBetaGammaFilter)

    def test_abg_full_name(self):
        abg = AutoFilter.from_filter(
            np.array([[0], [0], [0]]), 0.1, 0.1, 0.05, mode="alphabetagamma"
        )
        assert isinstance(abg, AlphaBetaGammaFilter)


# --- UKF modes ---


class TestAutoFilterUKF:
    def test_ukf_short(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x[:1]
        ukf = AutoFilter.from_filter(state, cov, Q, R, f, h, mode="ukf")
        assert isinstance(ukf, UnscentedKalmanFilter)

    def test_ukf_full_name(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x[:1]
        ukf = AutoFilter.from_filter(
            state, cov, Q, R, f, h, mode="UnscentedKalmanFilter"
        )
        assert isinstance(ukf, UnscentedKalmanFilter)


# --- PF modes ---


class TestAutoFilterPF:
    def test_pf_short(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x[:1]
        pf = AutoFilter.from_filter(state, cov, f, h, R, mode="pf")
        assert isinstance(pf, ParticleFilter)

    def test_pf_full_name(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x[:1]
        pf = AutoFilter.from_filter(state, cov, f, h, R, mode="particlefilter")
        assert isinstance(pf, ParticleFilter)

    def test_pf_alias(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x[:1]
        pf = AutoFilter.from_filter(state, cov, f, h, R, mode="particle")
        assert isinstance(pf, ParticleFilter)


# --- EnKF modes ---


class TestAutoFilterEnKF:
    def test_enkf_short(self):
        state = np.array([[0.0]])
        cov = np.eye(1)
        Q = np.eye(1) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x
        enkf = AutoFilter.from_filter(state, cov, Q, R, f, h, mode="enkf")
        assert isinstance(enkf, EnsembleKalmanFilter)

    def test_enkf_full_name(self):
        state = np.array([[0.0]])
        cov = np.eye(1)
        Q = np.eye(1) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x
        enkf = AutoFilter.from_filter(
            state, cov, Q, R, f, h, mode="EnsembleKalmanFilter"
        )
        assert isinstance(enkf, EnsembleKalmanFilter)

    def test_enkf_alias(self):
        state = np.array([[0.0]])
        cov = np.eye(1)
        Q = np.eye(1) * 0.01
        R = np.array([[0.1]])
        f = lambda x, dt: x
        h = lambda x: x
        enkf = AutoFilter.from_filter(state, cov, Q, R, f, h, mode="ensemble")
        assert isinstance(enkf, EnsembleKalmanFilter)


# --- IF modes ---


class TestAutoFilterIF:
    def test_if_short(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        filt = AutoFilter.from_filter(state, cov, F, Q, H, R, mode="if")
        assert isinstance(filt, InformationFilter)

    def test_if_full_name(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        filt = AutoFilter.from_filter(state, cov, F, Q, H, R, mode="InformationFilter")
        assert isinstance(filt, InformationFilter)

    def test_if_alias(self):
        state = np.array([[0.0], [0.0]])
        cov = np.eye(2)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        filt = AutoFilter.from_filter(state, cov, F, Q, H, R, mode="information")
        assert isinstance(filt, InformationFilter)


# --- AKF modes ---


class TestAutoFilterAKF:
    def test_akf_short(self, linear_args):
        akf = AutoFilter.from_filter(*linear_args, mode="akf")
        assert isinstance(akf, AdaptiveKalmanFilter)

    def test_akf_full_name(self, linear_args):
        akf = AutoFilter.from_filter(*linear_args, mode="AdaptiveKalmanFilter")
        assert isinstance(akf, AdaptiveKalmanFilter)

    def test_akf_alias(self, linear_args):
        akf = AutoFilter.from_filter(*linear_args, mode="adaptive")
        assert isinstance(akf, AdaptiveKalmanFilter)


# --- SRKF modes ---


class TestAutoFilterSRKF:
    def test_srkf_short(self, linear_args):
        srkf = AutoFilter.from_filter(*linear_args, mode="srkf")
        assert isinstance(srkf, SquareRootKalmanFilter)

    def test_srkf_full_name(self, linear_args):
        srkf = AutoFilter.from_filter(*linear_args, mode="SquareRootKalmanFilter")
        assert isinstance(srkf, SquareRootKalmanFilter)


# --- VKF modes ---


class TestAutoFilterVKF:
    def test_vkf_short(self):
        batch = 5
        state = np.zeros((batch, 2, 1))
        cov = np.repeat(np.eye(2)[np.newaxis, :, :], batch, axis=0)
        F = np.eye(2)
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        vkf = AutoFilter.from_filter(state, cov, F, Q, H, R, mode="vkf")
        assert isinstance(vkf, VectorizedKalmanFilter)

    def test_vkf_full_name(self):
        batch = 5
        state = np.zeros((batch, 2, 1))
        cov = np.repeat(np.eye(2)[np.newaxis, :, :], batch, axis=0)
        F = np.eye(2)
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        vkf = AutoFilter.from_filter(
            state, cov, F, Q, H, R, mode="VectorizedKalmanFilter"
        )
        assert isinstance(vkf, VectorizedKalmanFilter)


# --- Fading Memory KF modes ---


class TestAutoFilterFMKF:
    def test_fmkf_short(self, linear_args):
        fmkf = AutoFilter.from_filter(*linear_args, mode="fmkf")
        assert isinstance(fmkf, FadingMemoryKalmanFilter)

    def test_fmkf_full_name(self, linear_args):
        fmkf = AutoFilter.from_filter(*linear_args, mode="FadingMemoryKalmanFilter")
        assert isinstance(fmkf, FadingMemoryKalmanFilter)

    def test_fmkf_alias(self, linear_args):
        fmkf = AutoFilter.from_filter(*linear_args, mode="fading")
        assert isinstance(fmkf, FadingMemoryKalmanFilter)


# --- Invalid mode ---


class TestAutoFilterInvalid:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown filter mode"):
            AutoFilter.from_filter(mode="nonexistent")

    def test_invalid_mode_message(self):
        with pytest.raises(ValueError) as exc_info:
            AutoFilter.from_filter(mode="xyz")
        assert "kf, ekf, ukf" in str(exc_info.value)


# --- H-Infinity mode ---


class TestAutoFilterHInfinity:
    def test_hinf_short(self):
        kf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            mode="hinf",
        )
        assert isinstance(kf, HInfinityFilter)

    def test_hinf_full_name(self):
        kf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            mode="HInfinityFilter",
        )
        assert isinstance(kf, HInfinityFilter)

    def test_hinf_alias(self):
        kf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
            process_noise_cov=np.eye(2) * 0.01,
            measurement_matrix=np.array([[1.0, 0.0]]),
            measurement_noise_cov=np.array([[0.1]]),
            gamma=10.0,
            mode="hinfinity",
        )
        assert isinstance(kf, HInfinityFilter)


# --- SigmaPointUKF mode ---


class TestAutoFilterSPUKF:
    def test_spukf_short(self):
        ukf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            mode="spukf",
        )
        assert isinstance(ukf, SigmaPointUKF)

    def test_spukf_full_name(self):
        ukf = AutoFilter.from_filter(
            state=np.array([[0.0], [0.0]]),
            covariance=np.eye(2),
            transition_covariance=np.eye(2) * 0.01,
            measurement_covariance=np.array([[0.1]]),
            transition_function=lambda x, dt: x,
            measurement_function=lambda x: x[:1],
            mode="SigmaPointUKF",
        )
        assert isinstance(ukf, SigmaPointUKF)
