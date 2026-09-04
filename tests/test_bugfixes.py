import numpy as np
import pytest
from kalbee import (
    KalmanFilter,
    SquareRootKalmanFilter,
    FixedLagSmoother,
    HInfinityFilter,
    RaoBlackwellizedParticleFilter,
    track_to_track_fusion,
    plot_covariance,
)
from kalbee.modules.smoothers.rts_smoother import RTSSmoother


def test_srkf_multidim_equivalence():
    """SquareRootKalmanFilter must match standard KalmanFilter for m > 1 measurements."""
    state = np.array([[1.0], [2.0], [3.0]])
    covariance = np.array([
        [10.0, 2.0, 1.0],
        [2.0, 5.0, 0.5],
        [1.0, 0.5, 8.0]
    ])
    F = np.array([
        [1.2, 0.5, 0.0],
        [0.1, 0.9, 0.2],
        [0.0, 0.0, 1.0]
    ])
    Q = np.array([
        [0.5, 0.1, 0.0],
        [0.1, 0.3, 0.1],
        [0.0, 0.1, 0.4]
    ])
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    R = np.array([
        [0.5, 0.1],
        [0.1, 0.6]
    ])

    kf = KalmanFilter(state.copy(), covariance.copy(), F, Q, H, R)
    srkf = SquareRootKalmanFilter(state.copy(), covariance.copy(), F, Q, H, R)

    kf.predict()
    srkf.predict()
    np.testing.assert_allclose(kf.state, srkf.state, atol=1e-12)
    np.testing.assert_allclose(kf.covariance, srkf.covariance, atol=1e-12)

    z = np.array([[3.5], [1.8]])
    kf.update(z)
    srkf.update(z)

    np.testing.assert_allclose(kf.state, srkf.state, atol=1e-12)
    np.testing.assert_allclose(kf.covariance, srkf.covariance, atol=1e-12)


def test_fixed_lag_smoother_exact_rts_match():
    """FixedLagSmoother output must match full offline RTS Smoother."""
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])

    state = np.array([[0.0], [1.0]])
    cov = np.eye(2) * 10.0

    kf = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)
    smoother = FixedLagSmoother(kf, lag=3)

    measurements = [np.array([[float(t) + 0.1 * t]]) for t in range(5)]

    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []

    kf_ref = KalmanFilter(state.copy(), cov.copy(), F, Q, H, R)

    for z in measurements:
        kf_ref.predict()
        predicted_states.append(kf_ref.state.copy())
        predicted_covariances.append(kf_ref.covariance.copy())
        kf_ref.update(z)
        filtered_states.append(kf_ref.state.copy())
        filtered_covariances.append(kf_ref.covariance.copy())

        smoother.predict()
        smoother.update(z)

    rts_states, rts_covs = RTSSmoother.smooth(
        filtered_states, filtered_covariances, predicted_states, predicted_covariances, F
    )

    np.testing.assert_allclose(smoother.smoothed_state, rts_states[1], atol=1e-12)
    np.testing.assert_allclose(smoother.smoothed_covariance, rts_covs[1], atol=1e-12)


def test_correlated_track_to_track_fusion():
    """Correlated track fusion must execute Bar-Shalom / Campo formula cleanly."""
    mean_a = np.array([[1.0], [2.0]])
    cov_a = np.eye(2) * 2.0
    mean_b = np.array([[1.5], [2.2]])
    cov_b = np.eye(2) * 3.0
    cross_cov = np.array([[0.5, 0.1], [0.1, 0.4]])

    fused_mean, fused_cov = track_to_track_fusion(
        mean_a, cov_a, mean_b, cov_b, cross_covariance=cross_cov, method="klf"
    )

    assert fused_mean.shape == (2, 1)
    assert fused_cov.shape == (2, 2)
    # Fused covariance trace should be less than or equal to individual trace
    assert np.trace(fused_cov) < np.trace(cov_a)


def test_hinfinity_robust_covariance():
    """H-Infinity filter predict/update cycle."""
    kf = HInfinityFilter(
        state=np.array([[0.0], [0.0]]),
        covariance=np.eye(2) * 10.0,
        transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
        process_noise_cov=np.eye(2) * 0.01,
        measurement_matrix=np.array([[1.0, 0.0]]),
        measurement_noise_cov=np.array([[0.1]]),
        gamma=5.0,
    )
    kf.predict()
    state = kf.update(np.array([[1.0]]))
    assert state.shape == (2, 1)


def test_rbpf_linear_state_update():
    """RBPF linear state should be updated by measurement."""
    def f_linear(x_l, x_nl, dt):
        return np.array([[x_l[0, 0] + x_l[1, 0] * dt], [x_l[1, 0]]])

    def f_nonlinear(x_nl, dt):
        return np.array([[x_nl[0, 0] + x_nl[1, 0] * dt], [x_nl[1, 0]]])

    def h(x):
        return np.array([[x[0, 0] + x[2, 0]]])

    rbpf = RaoBlackwellizedParticleFilter(
        n_particles=20,
        linear_dim=2,
        nonlinear_dim=2,
        transition_function_linear=f_linear,
        transition_function_nonlinear=f_nonlinear,
        measurement_function=h,
        process_noise_linear=np.eye(2) * 0.01,
        process_noise_nonlinear=np.eye(2) * 0.01,
        measurement_noise=np.array([[0.1]]),
    )

    initial_linear_mean = rbpf.linear_means.mean(axis=0).copy()
    rbpf.predict(dt=1.0)
    rbpf.update(np.array([[5.0]]))

    # Linear states should have moved towards the measurement
    updated_linear_mean = rbpf.linear_means.mean(axis=0)
    assert not np.array_equal(initial_linear_mean, updated_linear_mean)


def test_plot_covariance_individual_states():
    """plot_covariance should return figure and axes without error."""
    pytest.importorskip("matplotlib")
    covs = [np.eye(2) * 1.0, np.eye(2) * 0.5]
    fig, ax = plot_covariance(covs, save_path=None)
    assert fig is not None
    assert ax is not None
