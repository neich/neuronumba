"""DSL-built models produce identical outputs to hand-written ones.

For each canonical model we compare:
  - the coupling kernel on a random coupling-state batch,
  - the dfun on a random (state, coupling) pair.
"""
import numpy as np

from neuronumba.simulator.models import (
    Hopf, Deco2014, Naskar2021, Montbrio,
    ZerlautAdaptationFirstOrder, ZerlautAdaptationSecondOrder,
)


def _compare_dfun(m_ref, m_dsl, n_rois, rng, atol=1e-12):
    state = rng.standard_normal((m_ref.n_state_vars, n_rois)) * 0.1
    coupling = rng.standard_normal((len(m_ref.c_vars), n_rois)) * 0.05

    f_ref = m_ref.get_numba_dfun()
    f_dsl = m_dsl.get_numba_dfun()
    ds_ref, obs_ref = f_ref(state.copy(), coupling.copy())
    ds_dsl, obs_dsl = f_dsl(state.copy(), coupling.copy())

    assert np.allclose(ds_ref, ds_dsl, atol=atol, rtol=atol)
    if obs_ref.size > 1 and obs_dsl.size > 1:
        assert obs_ref.shape == obs_dsl.shape
        assert np.allclose(obs_ref, obs_dsl, atol=atol, rtol=atol)


def _compare_coupling(m_ref, m_dsl, n_rois, rng, atol=1e-12):
    f_ref = m_ref.get_numba_coupling()
    f_dsl = m_dsl.get_numba_coupling()
    n_c = len(m_ref.c_vars)
    s = rng.standard_normal((n_c, n_rois)) * 0.1
    a = f_ref(s.copy())
    b = f_dsl(s.copy())
    assert np.allclose(a, b, atol=atol, rtol=atol)


def test_equivalence_hopf(weights, n_rois, hopf_dsl_cls):
    rng = np.random.default_rng(42)
    m_ref = Hopf(g=0.5).configure(weights=weights)
    m_dsl = hopf_dsl_cls(g=0.5).configure(weights=weights)
    _compare_coupling(m_ref, m_dsl, n_rois, rng)
    _compare_dfun(m_ref, m_dsl, n_rois, rng)


def test_equivalence_deco2014(weights, n_rois, deco_dsl_cls):
    rng = np.random.default_rng(42)
    m_ref = Deco2014(g=0.5, auto_fic=False).configure(weights=weights)
    m_dsl = deco_dsl_cls(g=0.5).configure(weights=weights)
    _compare_coupling(m_ref, m_dsl, n_rois, rng)
    _compare_dfun(m_ref, m_dsl, n_rois, rng)


def test_equivalence_naskar2021(weights, n_rois, naskar_dsl_cls):
    rng = np.random.default_rng(42)
    m_ref = Naskar2021(g=0.5).configure(weights=weights)
    m_dsl = naskar_dsl_cls(g=0.5).configure(weights=weights)
    _compare_coupling(m_ref, m_dsl, n_rois, rng)
    # Naskar's sigmoid has no eps-fix; tiny denominator differences can
    # propagate. 1e-10 still proves bit-equivalent math.
    _compare_dfun(m_ref, m_dsl, n_rois, rng, atol=1e-10)


def test_equivalence_montbrio(weights, n_rois, montbrio_dsl_cls):
    rng = np.random.default_rng(42)
    m_ref = Montbrio(g=0.5, auto_fic=False).configure(weights=weights)
    m_dsl = montbrio_dsl_cls(g=0.5).configure(weights=weights)
    _compare_coupling(m_ref, m_dsl, n_rois, rng)
    _compare_dfun(m_ref, m_dsl, n_rois, rng, atol=1e-10)


def test_equivalence_zerlaut_first_order(weights, n_rois, zerlaut_1o_dsl_cls):
    rng = np.random.default_rng(42)
    m_ref = ZerlautAdaptationFirstOrder(g=0.5).configure(weights=weights)
    m_dsl = zerlaut_1o_dsl_cls(g=0.5).configure(weights=weights)
    _compare_coupling(m_ref, m_dsl, n_rois, rng)
    _compare_dfun(m_ref, m_dsl, n_rois, rng)


def test_equivalence_zerlaut_second_order(weights, n_rois, zerlaut_2o_dsl_cls):
    rng = np.random.default_rng(42)
    m_ref = ZerlautAdaptationSecondOrder(g=0.5).configure(weights=weights)
    m_dsl = zerlaut_2o_dsl_cls(g=0.5).configure(weights=weights)
    _compare_coupling(m_ref, m_dsl, n_rois, rng)
    # 2nd-order's 12 chained TF calls + finite-diff derivatives sit at the FP
    # noise floor (~1e-12 absolute on dE/dI). The hand-written code keeps
    # inv_T as a 1D array (slice of `self.m`), while the DSL closes over it
    # as a scalar; numba's fastmath emits slightly different FMA sequences
    # for `arr * scalar` vs `arr * arr-of-constants`, so the lowest ~1 ulp
    # diverges per FLOP. 1e-11 covers that without hiding real bugs.
    _compare_dfun(m_ref, m_dsl, n_rois, rng, atol=1e-11)
