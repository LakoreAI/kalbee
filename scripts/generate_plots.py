import os
import numpy as np
import matplotlib.pyplot as plt

from kalbee import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
    EnsembleKalmanFilter,
    InformationFilter,
    AdaptiveKalmanFilter,
    SquareRootKalmanFilter,
    VectorizedKalmanFilter,
    InteractingMultipleModel,
    AlphaBetaGammaFilter,
)


def generate_all_plots():
    """
    Generate tracking illustration plots for each algorithm across three signal types:
    1. Sine/Cosine wave
    2. Degree 2 Polynomial
    3. Random Walk
    Using Standard KF as the baseline.
    """
    assets_dir = "docs/assets"
    os.makedirs(assets_dir, exist_ok=True)

    dt = 0.1
    t = np.arange(0, 10, dt)
    T = len(t)

    # 1. Define three ground truth signals and noisy measurements
    # Signal A: Sine/Cosine
    np.random.seed(42)
    noise_std = 0.4
    gt_sine = np.sin(t) + np.cos(t * 0.5)
    meas_sine = gt_sine + np.random.randn(T) * noise_std

    # Signal B: Polynomial degree 2 (y = 0.1 * t^2 - 0.5 * t + 1)
    gt_poly = 0.1 * (t**2) - 0.5 * t + 1.0
    meas_poly = gt_poly + np.random.randn(T) * noise_std

    # Signal C: Random Walk
    gt_rand = np.zeros(T)
    val = 0.0
    for i in range(T):
        val += np.random.randn() * 0.25
        gt_rand[i] = val
    meas_rand = gt_rand + np.random.randn(T) * noise_std

    signals = {
        "sine": (gt_sine, meas_sine),
        "poly": (gt_poly, meas_poly),
        "random": (gt_rand, meas_rand),
    }

    # Helper function to initialize baseline Standard KF
    def get_baseline_kf():
        state = np.zeros((2, 1))
        cov = np.eye(2)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.eye(2) * 0.05
        H = np.array([[1.0, 0.0]])
        R = np.eye(1) * (noise_std**2)
        return KalmanFilter(state, cov, F, Q, H, R)

    # Loop through each algorithm and generate plots
    algorithms = [
        "kf",
        "ekf",
        "ukf",
        "pf",
        "enkf",
        "if",
        "akf",
        "srkf",
        "vkf",
        "imm",
        "abg",
    ]

    for algo in algorithms:
        print(f"Generating plots for algorithm: {algo}...")
        for sig_name, (gt, meas) in signals.items():
            # Run Baseline KF
            baseline_kf = get_baseline_kf()
            est_baseline = []
            for i in range(T):
                baseline_kf.predict()
                baseline_kf.update(np.array([[meas[i]]]))
                est_baseline.append(baseline_kf.state[0, 0])

            # Run target algorithm
            est_algo = []
            state_init_2d = np.zeros((2, 1))
            cov_init_2d = np.eye(2)
            F_2d = np.array([[1.0, dt], [0.0, 1.0]])
            Q_2d = np.eye(2) * 0.05
            H_2d = np.array([[1.0, 0.0]])
            R_1d = np.eye(1) * (noise_std**2)

            if algo == "kf":
                algo_filter = KalmanFilter(
                    state_init_2d.copy(), cov_init_2d.copy(), F_2d, Q_2d, H_2d, R_1d
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "ekf":
                algo_filter = ExtendedKalmanFilter(
                    state=state_init_2d.copy(),
                    covariance=cov_init_2d.copy(),
                    transition_covariance=Q_2d,
                    measurement_covariance=R_1d,
                    transition_function=lambda s, d: F_2d @ s,
                    measurement_function=lambda s: H_2d @ s,
                    transition_jacobian=lambda s, d: F_2d,
                    measurement_jacobian=lambda s: H_2d,
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "ukf":
                algo_filter = UnscentedKalmanFilter(
                    state=state_init_2d.copy(),
                    covariance=cov_init_2d.copy(),
                    transition_covariance=Q_2d,
                    measurement_covariance=R_1d,
                    transition_function=lambda s, d: F_2d @ s,
                    measurement_function=lambda s: H_2d @ s,
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "pf":
                algo_filter = ParticleFilter(
                    state=state_init_2d.copy(),
                    covariance=cov_init_2d.copy(),
                    transition_function=lambda p, d: (
                        F_2d @ p + np.random.randn(*p.shape) * 0.1
                    ),
                    measurement_function=lambda p: H_2d @ p,
                    measurement_covariance=R_1d,
                    num_particles=150,
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([meas[i]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "enkf":
                algo_filter = EnsembleKalmanFilter(
                    state=state_init_2d.copy(),
                    covariance=cov_init_2d.copy(),
                    transition_covariance=Q_2d,
                    measurement_covariance=R_1d,
                    transition_function=lambda s, d: F_2d @ s,
                    measurement_function=lambda s: H_2d @ s,
                    ensemble_size=30,
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "if":
                algo_filter = InformationFilter(
                    state_init_2d.copy(), cov_init_2d.copy(), F_2d, Q_2d, H_2d, R_1d
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "akf":
                algo_filter = AdaptiveKalmanFilter(
                    state_init_2d.copy(), cov_init_2d.copy(), F_2d, Q_2d, H_2d, R_1d
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "srkf":
                algo_filter = SquareRootKalmanFilter(
                    state_init_2d.copy(), cov_init_2d.copy(), F_2d, Q_2d, H_2d, R_1d
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "vkf":
                # VKF processes a batch of 1
                state_batch = state_init_2d.copy()[np.newaxis, :, :]
                cov_batch = cov_init_2d.copy()[np.newaxis, :, :]
                F_batch = F_2d[np.newaxis, :, :]
                Q_batch = Q_2d[np.newaxis, :, :]
                H_batch = H_2d[np.newaxis, :, :]
                R_batch = R_1d[np.newaxis, :, :]
                algo_filter = VectorizedKalmanFilter(
                    state_batch, cov_batch, F_batch, Q_batch, H_batch, R_batch
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[[meas[i]]]]))
                    est_algo.append(algo_filter.state[0, 0, 0])

            elif algo == "imm":
                # IMM blends CV and CA models (using matching 3D state representation for 1D signals)
                state_ca = np.array([[0.0], [0.0], [0.0]])
                cov_ca = np.eye(3) * 2.0
                F_cv_3 = np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
                Q_cv_3 = np.zeros((3, 3))
                Q_cv_3[:2, :2] = Q_2d
                F_ca_3 = np.array(
                    [[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]
                )
                Q_ca_3 = np.eye(3) * 0.1
                H_ca_3 = np.array([[1.0, 0.0, 0.0]])

                kf_cv_imm = KalmanFilter(
                    state_ca.copy(), cov_ca.copy(), F_cv_3, Q_cv_3, H_ca_3, R_1d.copy()
                )
                kf_ca_imm = KalmanFilter(
                    state_ca.copy(), cov_ca.copy(), F_ca_3, Q_ca_3, H_ca_3, R_1d.copy()
                )
                algo_filter = InteractingMultipleModel(
                    [kf_cv_imm, kf_ca_imm],
                    np.array([[0.95, 0.05], [0.05, 0.95]]),
                    np.array([0.8, 0.2]),
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            elif algo == "abg":
                algo_filter = AlphaBetaGammaFilter(
                    state=np.array([[0.0], [0.0], [0.0]]),
                    alpha=0.4,
                    beta=0.1,
                    gamma=0.01,
                )
                for i in range(T):
                    algo_filter.predict()
                    algo_filter.update(np.array([[meas[i]]]))
                    est_algo.append(algo_filter.state[0, 0])

            # Generate Plot
            plt.figure(figsize=(10, 6))
            plt.plot(t, gt, "g-", label="Ground Truth", linewidth=2.0)
            plt.scatter(
                t, meas, color="red", alpha=0.3, s=15, label="Noisy Measurements"
            )
            plt.plot(t, est_baseline, "b--", label="Standard KF Baseline", alpha=0.7)
            plt.plot(t, est_algo, "m-", label=f"{algo.upper()} Estimate", linewidth=1.5)

            plt.title(
                f"Tracking Comparison: {algo.upper()} on {sig_name.capitalize()} Signal"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Position")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()

            # Save Image
            save_path = os.path.join(assets_dir, f"{algo}_{sig_name}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close()

    print("All tracking illustration plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
