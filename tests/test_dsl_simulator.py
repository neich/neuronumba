"""End-to-end equivalence: simulator trajectories match hand-written models.

Both runs use `EulerDeterministic` so the only source of variation is the dfun
and coupling kernel — both already proved bit-equal in test_dsl_equivalence.
We seed numpy globally before each `simulate_nodelay` call because that helper
draws random tract lengths internally; HistoryNoDelays ignores lengths but we
still seed for reproducibility.
"""
import numpy as np

from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.simulator.integrators.euler import EulerDeterministic
from neuronumba.simulator.models import Hopf, Deco2014


def _run(model_cls, weights, obs_var, *, dt=0.1, t_max=200.0, **model_kwargs):
    np.random.seed(0)
    model = model_cls(**model_kwargs)
    integrator = EulerDeterministic(dt=dt)
    return simulate_nodelay(
        model, integrator, weights, obs_var,
        sampling_period=1.0,
        t_max_neuronal=t_max,
        t_warmup=0.0,
    )


def test_simulator_end_to_end_hopf_deterministic(weights, hopf_dsl_cls):
    ref = _run(Hopf, weights, obs_var="x", g=0.5)
    dsl = _run(hopf_dsl_cls, weights, obs_var="x", g=0.5)
    assert ref.shape == dsl.shape
    assert np.allclose(ref, dsl, atol=1e-12, rtol=1e-12)


def test_simulator_end_to_end_deco2014_deterministic(weights, deco_dsl_cls):
    ref = _run(Deco2014, weights, obs_var="S_e", g=0.5, auto_fic=False)
    dsl = _run(deco_dsl_cls, weights, obs_var="S_e", g=0.5)
    assert ref.shape == dsl.shape
    assert np.allclose(ref, dsl, atol=1e-12, rtol=1e-12)
