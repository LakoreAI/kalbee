import numpy as np

from kalbee.modules.utils.jacobian import (
    numerical_jacobian,
    numerical_transition_jacobian,
    numerical_measurement_jacobian,
)


def test_numerical_jacobian_of_linear_function_matches_matrix():
    A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
    x = np.array([[1.0], [2.0], [3.0]])

    J = numerical_jacobian(lambda s: A @ s, x)

    assert np.allclose(J, A, atol=1e-6)


def test_numerical_jacobian_of_nonlinear_function():
    def f(x):
        return np.array([[x[0, 0] ** 2], [x[0, 0] * x[1, 0]]])

    x = np.array([[2.0], [3.0]])
    J = numerical_jacobian(f, x)

    expected = np.array([[2 * x[0, 0], 0.0], [x[1, 0], x[0, 0]]])
    assert np.allclose(J, expected, atol=1e-5)


def test_numerical_transition_jacobian_matches_linear_F():
    dt = 0.5
    F = np.array([[1.0, dt], [0.0, 1.0]])

    def f(x, dt):
        return F @ x

    x = np.array([[1.0], [2.0]])
    J = numerical_transition_jacobian(f, x, dt)
    assert np.allclose(J, F, atol=1e-6)


def test_numerical_measurement_jacobian_matches_linear_H():
    H = np.array([[1.0, 0.0]])

    def h(x):
        return H @ x

    x = np.array([[3.0], [4.0]])
    J = numerical_measurement_jacobian(h, x)
    assert np.allclose(J, H, atol=1e-6)
