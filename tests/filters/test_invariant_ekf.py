"""Tests for the Invariant EKF on SO(3)/SE(3)."""

import numpy as np

from kalbee import InvariantEKF, SE3, SO3


def test_so3_se3_exp_log():
    """Test Lie Group SO(3) and SE(3) matrix exp/log functions."""
    w = np.array([0.1, 0.2, 0.3])
    R = SO3.exp(w)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    w_rec = SO3.log(R).flatten()
    np.testing.assert_allclose(w, w_rec, atol=1e-5)

    xi = np.array([1.0, 2.0, 3.0, 0.1, 0.0, 0.0])
    T = SE3.exp(xi)
    assert T.shape == (4, 4)
    assert SE3.adjoint(T).shape == (6, 6)


def test_invariant_ekf():
    """InvariantEKF should predict and update SE(3) pose cleanly."""
    in_ekf = InvariantEKF()
    assert in_ekf.pose.shape == (4, 4)
    assert in_ekf.rotation.shape == (3, 3)
    assert in_ekf.position.shape == (3, 1)

    twist = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    in_ekf.predict(dt=1.0, twist=twist)
    assert in_ekf.pose.shape == (4, 4)

    z = np.array([[1.0], [0.1], [0.0]])
    in_ekf.update(z)
    assert in_ekf.pose.shape == (4, 4)
