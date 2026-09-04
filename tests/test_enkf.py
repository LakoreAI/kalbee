import numpy as np
from kalbee.modules.filters.enkf_filter import EnsembleKalmanFilter


def test_enkf_initialization():
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2)
    Q = np.eye(2) * 0.1
    R = np.eye(1) * 0.5

    def f(x, dt):
        return x

    def h(x):
        return x[:1]

    enkf = EnsembleKalmanFilter(
        state=state,
        covariance=cov,
        transition_covariance=Q,
        measurement_covariance=R,
        transition_function=f,
        measurement_function=h,
        ensemble_size=50,
    )

    assert enkf.ensemble_size == 50
    assert enkf.ensemble.shape == (50, 2)


def test_enkf_predict_linear():
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 0.01
    Q = np.zeros((1, 1))
    R = np.eye(1)

    # f(x) = x + dt
    def transition(x, dt):
        return x + dt

    def measurement(x):
        return x

    enkf = EnsembleKalmanFilter(
        state=state,
        covariance=cov,
        transition_covariance=Q,
        measurement_covariance=R,
        transition_function=transition,
        measurement_function=measurement,
        ensemble_size=200,
    )

    enkf.predict(dt=1.0)

    # Mean should be approximately 1.0
    assert np.isclose(enkf.state[0, 0], 1.0, atol=0.2)


def test_enkf_update_convergence():
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 10.0
    Q = np.eye(1) * 0.01
    R = np.eye(1) * 0.1

    def transition(x, dt):
        return x

    def measurement(x):
        return x

    enkf = EnsembleKalmanFilter(
        state=state,
        covariance=cov,
        transition_covariance=Q,
        measurement_covariance=R,
        transition_function=transition,
        measurement_function=measurement,
        ensemble_size=200,
    )

    # Several measurements at 5.0
    for _ in range(5):
        enkf.predict(dt=1.0)
        enkf.update(np.array([[5.0]]))

    # State should converge toward 5.0
    assert np.isclose(enkf.state[0, 0], 5.0, atol=1.0)


def test_enkf_covariance_decreases():
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 10.0
    Q = np.eye(1) * 0.01
    R = np.eye(1) * 0.1

    def transition(x, dt):
        return x

    def measurement(x):
        return x

    enkf = EnsembleKalmanFilter(
        state=state,
        covariance=cov,
        transition_covariance=Q,
        measurement_covariance=R,
        transition_function=transition,
        measurement_function=measurement,
        ensemble_size=200,
    )

    initial_cov = enkf.covariance[0, 0]

    enkf.predict(dt=1.0)
    enkf.update(np.array([[3.0]]))

    # Covariance should decrease after update
    assert enkf.covariance[0, 0] < initial_cov
