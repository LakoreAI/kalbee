import numpy as np
from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.imm_filter import InteractingMultipleModel
from kalbee.modules.utils.metrics import rmse


def run_tracking_simulation():
    """
    Simulate a 2D maneuvering target and track it using CV Kalman Filter,
    CA Kalman Filter, and an IMM blending both.
    """
    print("Initializing maneuvering target simulation...")
    # Time parameters
    dt = 0.1
    t = np.arange(0, 15, dt)
    T = len(t)

    # 1. Generate maneuvering target trajectory
    # CV model first (0 to 5s), then high acceleration (5s to 10s), then CV again (10s to 15s)
    true_x = []
    true_v = []
    true_a = []

    x = 0.0
    v = 2.0
    a = 0.0

    for step in t:
        if 5.0 <= step < 10.0:
            a = 1.5  # acceleration phase
        else:
            a = 0.0  # constant velocity phase

        v += a * dt
        x += v * dt + 0.5 * a * dt**2

        true_x.append(x)
        true_v.append(v)
        true_a.append(a)

    true_x = np.array(true_x)

    # Noisy measurements
    np.random.seed(42)
    noise_std = 0.5
    measurements = true_x + np.random.randn(T) * noise_std

    # 2. Initialize CV Filter (Constant Velocity)
    # State: [position, velocity]^T
    state_cv = np.array([[0.0], [2.0]])
    cov_cv = np.eye(2) * 5.0
    F_cv = np.array([[1.0, dt], [0.0, 1.0]])
    Q_cv = (
        np.array(
            [
                [dt**4 / 4, dt**3 / 2],
                [dt**3 / 2, dt**2],
            ]
        )
        * 0.01
    )
    H_cv = np.array([[1.0, 0.0]])
    R_cv = np.array([[noise_std**2]])

    kf_cv = KalmanFilter(state_cv.copy(), cov_cv.copy(), F_cv, Q_cv, H_cv, R_cv)

    # 3. Initialize CA Filter (Constant Acceleration)
    # State: [position, velocity, acceleration]^T
    state_ca = np.array([[0.0], [2.0], [0.0]])
    cov_ca = np.eye(3) * 5.0
    F_ca = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    Q_ca = (
        np.array(
            [
                [dt**6 / 36, dt**5 / 12, dt**4 / 6],
                [dt**5 / 12, dt**4 / 4, dt**3 / 2],
                [dt**4 / 6, dt**3 / 2, dt**2],
            ]
        )
        * 0.1
    )
    H_ca = np.array([[1.0, 0.0, 0.0]])
    R_ca = np.array([[noise_std**2]])

    kf_ca = KalmanFilter(state_ca.copy(), cov_ca.copy(), F_ca, Q_ca, H_ca, R_ca)

    # 4. Initialize IMM blending CV and CA
    # To run in IMM, the filters must have matching state dimensions.
    # We can design a 3D state IMM where the CV model has the acceleration state fixed to 0.
    # So we define the CV filter with state dimension 3.
    F_cv3 = np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    Q_cv3 = np.zeros((3, 3))
    Q_cv3[:2, :2] = Q_cv
    H_cv3 = np.array([[1.0, 0.0, 0.0]])

    kf_cv_imm = KalmanFilter(
        state_ca.copy(),
        cov_ca.copy(),
        F_cv3,
        Q_cv3,
        H_cv3,
        R_ca.copy(),
    )
    kf_ca_imm = KalmanFilter(
        state_ca.copy(),
        cov_ca.copy(),
        F_ca,
        Q_ca,
        H_ca,
        R_ca.copy(),
    )

    model_transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    model_probabilities = np.array([0.8, 0.2])  # Prefer CV initially

    imm = InteractingMultipleModel(
        [kf_cv_imm, kf_ca_imm],
        model_transition,
        model_probabilities,
    )

    # 5. Run simulation loop
    est_cv = []
    est_ca = []
    est_imm = []
    probs_cv = []
    probs_ca = []

    for step in range(T):
        z = np.array([[measurements[step]]])

        # Step CV filter
        kf_cv.predict()
        kf_cv.update(z)
        est_cv.append(kf_cv.state[0, 0])

        # Step CA filter
        kf_ca.predict()
        kf_ca.update(z)
        est_ca.append(kf_ca.state[0, 0])

        # Step IMM filter
        imm.predict()
        imm.update(z)
        est_imm.append(imm.state[0, 0])

        # Record probabilities
        probs_cv.append(imm.model_probabilities[0])
        probs_ca.append(imm.model_probabilities[1])

    est_cv = np.array(est_cv)
    est_ca = np.array(est_ca)
    est_imm = np.array(est_imm)

    # 6. Print Performance Metrics
    print("\nTracking Performance (Position RMSE):")
    print(f"Constant Velocity (CV) KF:  {rmse(est_cv, true_x):.4f}")
    print(f"Constant Acceleration (CA) KF: {rmse(est_ca, true_x):.4f}")
    print(f"IMM Blended Filter:            {rmse(est_imm, true_x):.4f}")

    print("\nIMM Adaptation Timeline:")
    print(
        f"Start (CV Phase): CV Probability = {probs_cv[0]:.2f}, CA Probability = {probs_ca[0]:.2f}"
    )
    mid_index = int(T / 2)
    print(
        f"Maneuver (CA Phase): CV Probability = {probs_cv[mid_index]:.2f}, CA Probability = {probs_ca[mid_index]:.2f}"
    )
    print(
        f"End (CV Phase):   CV Probability = {probs_cv[-1]:.2f}, CA Probability = {probs_ca[-1]:.2f}"
    )


if __name__ == "__main__":
    run_tracking_simulation()
