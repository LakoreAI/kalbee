import numpy as np

from kalbee.modules.filters.abg_filter import AlphaBetaGammaFilter


def test_abg_initialization():
    state = np.array([[0], [1], [0.1]])
    alpha, beta, gamma = 0.5, 0.4, 0.1

    abg = AlphaBetaGammaFilter(state, alpha, beta, gamma)

    assert np.array_equal(abg.state, state)
    assert abg.alpha == alpha
    assert abg.beta == beta
    assert abg.gamma == gamma
    # Base class initialization check
    assert abg.covariance is not None


def test_abg_predict():
    state = np.array([[0.0], [1.0], [0.0]])  # Position 0, Velocity 1, Accel 0
    alpha, beta, gamma = 0.5, 0.4, 0.1
    dt = 1.0

    abg = AlphaBetaGammaFilter(state, alpha, beta, gamma)
    abg.predict(dt=dt)

    # Expected: x = 0 + 1*1 + 0 = 1
    # v = 1 + 0 = 1
    # a = 0
    expected_state = np.array([[1.0], [1.0], [0.0]])
    assert np.allclose(abg.state, expected_state)


def test_abg_update():
    state = np.array([[0.0], [0.0], [0.0]])
    alpha, beta, gamma = 1.0, 1.0, 1.0  # Full correction for test
    dt = 1.0

    abg = AlphaBetaGammaFilter(state, alpha, beta, gamma)

    # Predict step (stays 0)
    abg.predict(dt=dt)

    # Measurement at 10.0
    abg.update(np.array([[10.0]]), dt=dt)

    # x = 0 + 1.0 * (10 - 0) = 10
    # v = 0 + (1.0/1.0) * (10 - 0) = 10
    # a = 0 + (1.0/(2*1^2)) * (10 - 0) = 5
    assert abg.state[0, 0] == 10.0
    assert abg.state[1, 0] == 10.0
    assert abg.state[2, 0] == 5.0
