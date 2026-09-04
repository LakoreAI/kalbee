import numpy as np
from kalbee.modules.filters.particle_filter import ParticleFilter


def test_pf_initialization():
    state = np.array([[0.0], [0.0]])
    cov = np.eye(2)
    R = np.eye(1) * 0.5

    def f(x, dt):
        return x

    def h(x):
        return x[:1]

    pf = ParticleFilter(
        state=state,
        covariance=cov,
        transition_function=f,
        measurement_function=h,
        measurement_covariance=R,
        num_particles=100,
    )

    assert pf.num_particles == 100
    assert pf.particles.shape == (100, 2)
    assert np.isclose(np.sum(pf.weights), 1.0)


def test_pf_predict():
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 0.01
    R = np.eye(1) * 0.1

    # Constant velocity: x = x + dt
    def transition(x, dt):
        return x + dt

    def measurement(x):
        return x

    pf = ParticleFilter(
        state=state,
        covariance=cov,
        transition_function=transition,
        measurement_function=measurement,
        measurement_covariance=R,
        num_particles=1000,
    )

    pf.predict(dt=1.0)

    # Mean should be approximately 1.0 (starting at 0 + 1*dt)
    assert np.isclose(pf.state[0, 0], 1.0, atol=0.5)


def test_pf_update_convergence():
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 10.0  # High initial uncertainty
    R = np.eye(1) * 0.01  # Low measurement noise

    def transition(x, dt):
        return x

    def measurement(x):
        return x

    pf = ParticleFilter(
        state=state,
        covariance=cov,
        transition_function=transition,
        measurement_function=measurement,
        measurement_covariance=R,
        num_particles=2000,
    )

    # Feed several measurements at 5.0
    for _ in range(10):
        pf.predict(dt=1.0)
        pf.update(np.array([[5.0]]))

    # Should converge close to 5.0
    assert np.isclose(pf.state[0, 0], 5.0, atol=1.0)


def test_pf_resampling():
    np.random.seed(42)
    state = np.array([[0.0]])
    cov = np.eye(1) * 1.0
    R = np.eye(1) * 0.1

    def transition(x, dt):
        return x

    def measurement(x):
        return x

    pf = ParticleFilter(
        state=state,
        covariance=cov,
        transition_function=transition,
        measurement_function=measurement,
        measurement_covariance=R,
        num_particles=100,
        resample_threshold=0.5,
    )

    pf.predict(dt=1.0)
    pf.update(np.array([[3.0]]))

    # After resampling, weights should be uniform
    assert np.isclose(np.sum(pf.weights), 1.0)
    assert pf._effective_particles() > 0


# ============================================================
# RNG reproducibility and vectorized vs looped equivalence
# ============================================================


def _cv_funcs():
    def f(x, dt):
        return x + dt  # scalar constant-velocity-ish, works for (n,1) and (n,N)

    def h(x):
        return x[:1]

    return f, h


def test_pf_rng_reproducible():
    f, h = _cv_funcs()
    kwargs = dict(
        state=np.array([[0.0]]),
        covariance=np.eye(1),
        transition_function=f,
        measurement_function=h,
        measurement_covariance=np.eye(1) * 0.1,
        num_particles=200,
    )
    pf_a = ParticleFilter(rng=123, **kwargs)
    pf_b = ParticleFilter(rng=123, **kwargs)
    assert np.allclose(pf_a.particles, pf_b.particles)


def test_pf_rng_does_not_touch_global_state():
    # Migrating off np.random.seed means constructing a filter must not perturb
    # the global RNG stream.
    np.random.seed(7)
    before = np.random.rand()
    np.random.seed(7)
    f, h = _cv_funcs()
    ParticleFilter(
        state=np.array([[0.0]]),
        covariance=np.eye(1),
        transition_function=f,
        measurement_function=h,
        measurement_covariance=np.eye(1) * 0.1,
        num_particles=200,
        rng=999,
    )
    after = np.random.rand()
    assert before == after


def test_pf_vectorized_matches_looped():
    # Linear CV funcs support both per-particle and column-batched input, so the
    # two code paths must agree given the same rng.
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    def f(x, dt):
        return F @ x

    def h(x):
        return H @ x

    kwargs = dict(
        state=np.zeros((2, 1)),
        covariance=np.eye(2),
        transition_function=f,
        measurement_function=h,
        measurement_covariance=np.array([[0.2]]),
        num_particles=300,
    )
    pf_loop = ParticleFilter(rng=5, **kwargs)
    pf_vec = ParticleFilter(rng=5, vectorized_functions=True, **kwargs)

    for _ in range(5):
        pf_loop.predict(dt=1.0)
        pf_vec.predict(dt=1.0)
        z = np.array([[1.0]])
        pf_loop.update(z)
        pf_vec.update(z)

    assert np.allclose(pf_loop.state, pf_vec.state)
