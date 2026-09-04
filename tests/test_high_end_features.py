import numpy as np
from kalbee import (
    InvariantEKF,
    SO3,
    SE3,
    VariationalBayesKalmanFilter,
)
from kalbee.tracking.pmbm import PMBMTracker
from kalbee.modules.integration.factor_graph import FactorGraphExporter


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


def test_variational_bayes_kalman_filter():
    """VBAKF should estimate state and adjust measurement covariance R online."""
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2) * 10.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R_init = np.array([[1.0]])

    vbakf = VariationalBayesKalmanFilter(state, cov, F, Q, H, R_init, n_iter=3)

    for t in range(5):
        vbakf.predict()
        vbakf.update(np.array([[float(t + 1)]]))

    assert vbakf.state.shape == (2, 1)
    assert vbakf.measurement_covariance.shape == (1, 1)


def test_kalmannet():
    """KalmanNet should run PyTorch step when torch is available."""
    try:
        import torch
        from kalbee.modules.learning.kalmannet import KalmanNet

        knet = KalmanNet(state_dim=2, meas_dim=1, hidden_dim=16)
        x_pred = torch.zeros(1, 2, 1)
        z = torch.ones(1, 1, 1)
        H = torch.tensor([[1.0, 0.0]]).unsqueeze(0)
        h_rnn = torch.zeros(1, 16)

        x_upd, h_next = knet.step(x_pred, z, H, h_rnn)
        assert x_upd.shape == (1, 2, 1)
        assert h_next.shape == (1, 16)
    except ImportError:
        pass


def test_pmbm_tracker():
    """PMBMTracker should track target hypotheses and manage births/pruning."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])

    pmbm = PMBMTracker(F, Q, H, R, birth_rate=0.1)

    # Frame 1: 1 detection
    pmbm.predict(dt=1.0)
    confirmed = pmbm.update(np.array([[1.0]]))

    # Frame 2: detection moves
    pmbm.predict(dt=1.0)
    confirmed = pmbm.update(np.array([[2.0]]))

    assert isinstance(confirmed, list)


def test_factor_graph_exporter(tmp_path):
    """FactorGraphExporter should output graph nodes/factors as JSON."""
    exporter = FactorGraphExporter()
    exporter.add_state_node(0, np.zeros((2, 1)), np.eye(2))
    exporter.add_motion_factor(0, 1, np.eye(2), np.eye(2) * 0.01)
    exporter.add_measurement_factor(0, np.array([[1.0]]), np.array([[1, 0]]), np.array([[0.1]]))

    graph_dict = exporter.to_dict()
    assert len(graph_dict["nodes"]) == 1
    assert len(graph_dict["factors"]) == 2

    filepath = tmp_path / "graph.json"
    exporter.save_json(str(filepath))
    assert filepath.exists()
