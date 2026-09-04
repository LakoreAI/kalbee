import numpy as np

from kalbee.modules.filters.particle_filter import ParticleFilter
from kalbee.modules.filters.enkf_filter import EnsembleKalmanFilter
from kalbee.experiments.signals import sine_signal


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


def test_enkf_vectorized_matches_looped():
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    def f(x, dt):
        return F @ x

    def h(x):
        return H @ x

    kwargs = dict(
        state=np.zeros((2, 1)),
        covariance=np.eye(2),
        transition_covariance=np.eye(2) * 0.01,
        measurement_covariance=np.array([[0.2]]),
        transition_function=f,
        measurement_function=h,
        ensemble_size=80,
    )
    en_loop = EnsembleKalmanFilter(rng=3, **kwargs)
    en_vec = EnsembleKalmanFilter(rng=3, vectorized_functions=True, **kwargs)

    for _ in range(5):
        en_loop.predict(dt=1.0)
        en_vec.predict(dt=1.0)
        z = np.array([[1.0]])
        en_loop.update(z)
        en_vec.update(z)

    assert np.allclose(en_loop.state, en_vec.state)


def test_signal_seed_reproducible_and_isolated():
    _, _, m1 = sine_signal(duration=2.0, dt=0.1, seed=42)
    _, _, m2 = sine_signal(duration=2.0, dt=0.1, seed=42)
    assert np.allclose(m1, m2)

    # Passing a Generator is also supported.
    gen = np.random.default_rng(0)
    _, _, m3 = sine_signal(duration=2.0, dt=0.1, seed=gen)
    assert np.all(np.isfinite(m3))
