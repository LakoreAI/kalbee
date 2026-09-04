import numpy as np

from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.imm_filter import InteractingMultipleModel


def test_imm_initialization():
    state1 = np.array([[1.0], [0.0]])
    cov1 = np.eye(2)
    F1 = np.eye(2)
    Q1 = np.eye(2) * 0.1
    H1 = np.array([[1.0, 0.0]])
    R1 = np.eye(1)

    kf1 = KalmanFilter(state1, cov1, F1, Q1, H1, R1)

    state2 = np.array([[2.0], [0.0]])
    cov2 = np.eye(2)
    kf2 = KalmanFilter(state2, cov2, F1, Q1, H1, R1)

    filters = [kf1, kf2]
    # Transition probability: very likely to stay in same model
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    # Start with equal probabilities
    model_probabilities = np.array([0.5, 0.5])

    imm = InteractingMultipleModel(filters, transition_matrix, model_probabilities)

    # Initial combined state: 0.5 * [1, 0] + 0.5 * [2, 0] = [1.5, 0.0]
    expected_state = np.array([[1.5], [0.0]])
    assert np.allclose(imm.state, expected_state)
    assert np.allclose(imm.x, expected_state)
    assert np.array_equal(imm.model_probabilities, np.array([0.5, 0.5]))


def test_imm_predict_update():
    # Model 1: Slow process noise
    kf1 = KalmanFilter(
        state=np.array([[0.0], [1.0]]),
        covariance=np.eye(2),
        transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
        transition_covariance=np.eye(2) * 0.01,
        measurement_matrix=np.array([[1.0, 0.0]]),
        measurement_covariance=np.array([[0.1]]),
    )

    # Model 2: Fast process noise
    kf2 = KalmanFilter(
        state=np.array([[0.0], [1.0]]),
        covariance=np.eye(2),
        transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]]),
        transition_covariance=np.eye(2) * 2.0,
        measurement_matrix=np.array([[1.0, 0.0]]),
        measurement_covariance=np.array([[0.1]]),
    )

    filters = [kf1, kf2]
    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
    model_probabilities = np.array([0.8, 0.2])

    imm = InteractingMultipleModel(filters, transition_matrix, model_probabilities)

    # Predict
    imm.predict()

    # Model probabilities after prediction (prior) should be transitioned
    # c = P_trans.T @ mu = [[0.95, 0.05], [0.05, 0.95]] @ [0.8, 0.2]
    # c_1 = 0.95 * 0.8 + 0.05 * 0.2 = 0.76 + 0.01 = 0.77
    # c_2 = 0.05 * 0.8 + 0.95 * 0.2 = 0.04 + 0.19 = 0.23
    assert np.allclose(imm.model_probabilities, np.array([0.77, 0.23]))

    # Update with measurement matching model 1 (slow)
    # Predicted state position is ~1.0
    imm.update(np.array([[1.1]]))

    # Since measurement 1.1 is very close to slow model prediction, model 1 should remain highly probable
    assert imm.model_probabilities[0] > 0.5
