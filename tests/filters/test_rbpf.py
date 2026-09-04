import numpy as np

from kalbee import RaoBlackwellizedParticleFilter


class TestRaoBlackwellizedParticleFilter:
    """Tests for RBPF."""

    def test_basic_rbpf(self):
        def f_linear(x_l, x_nl, dt):
            return np.array([[x_l[0, 0] + x_l[1, 0] * dt], [x_l[1, 0]]])

        def f_nonlinear(x_nl, dt):
            return np.array([[x_nl[0, 0] + np.sin(x_nl[1, 0]) * dt], [x_nl[1, 0]]])

        def h(x):
            return np.array([[x[0, 0]]])

        rbpf = RaoBlackwellizedParticleFilter(
            n_particles=50,
            linear_dim=2,
            nonlinear_dim=2,
            transition_function_linear=f_linear,
            transition_function_nonlinear=f_nonlinear,
            measurement_function=h,
            process_noise_linear=np.eye(2) * 0.01,
            process_noise_nonlinear=np.eye(2) * 0.01,
            measurement_noise=np.array([[0.5]]),
        )

        rbpf.predict(dt=1.0)
        rbpf.update(np.array([[1.0]]))

        assert rbpf.state.shape == (4, 1)
        assert np.isclose(rbpf.weights.sum(), 1.0)

    def test_linear_state_update(self):
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
