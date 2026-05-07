"""Dependent-parameter analysis and runtime evaluation."""
import numpy as np
import pytest

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


def _spec_with_params(params, name="DepModel", equations="d_x = -x + coupling.x"):
    return ModelSpec(
        name=name,
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=params,
        equations=equations,
    )


def test_dependents_simple_chain():
    spec = _spec_with_params(
        [
            Parameter("a", default=2.0),
            Parameter("b", default=3.0),
            Parameter("c", formula="a + b"),
        ],
        name="ChainDep",
        equations="d_x = -x / c + coupling.x",
    )
    Cls = build_model(spec)
    n = 4
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))
    assert np.allclose(m.c, 5.0)
    assert np.allclose(m.m[int(m.P.c)], 5.0)


def test_dependents_multilevel_topo_sort():
    spec = _spec_with_params(
        [
            Parameter("a", default=2.0),
            Parameter("b", default=3.0),
            Parameter("c", default=4.0),
            Parameter("d", formula="a + b"),
            Parameter("e", formula="c * d"),
            Parameter("f", formula="np.exp(0) * e"),
        ],
        name="MultiDep",
        equations="d_x = -x + f * coupling.x",
    )
    Cls = build_model(spec)
    m = Cls(g=0.0).configure(weights=np.zeros((3, 3)))
    assert np.allclose(m.d, 5.0)
    assert np.allclose(m.e, 20.0)
    assert np.allclose(m.f, 20.0)


def test_dependents_montbrio_pattern():
    """The recurrent Montbrio pattern: J_N_ee = J_ee + g_ee * log(a_e)."""
    spec = _spec_with_params(
        [
            Parameter("J_ee",   default=10.0),
            Parameter("g_ee",   default=2.5),
            Parameter("a_e",    default=0.25),
            Parameter("J_N_ee", formula="J_ee + g_ee * np.log(a_e)"),
        ],
        name="MontbrioDep",
        equations="d_x = -x * J_N_ee + coupling.x",
    )
    Cls = build_model(spec)
    m = Cls(g=0.0).configure(weights=np.zeros((3, 3)))
    expected = 10.0 + 2.5 * np.log(0.25)
    assert np.allclose(m.J_N_ee, expected)


def test_dependents_cycle_detected():
    spec = _spec_with_params(
        [
            Parameter("a", formula="b + 1"),
            Parameter("b", formula="a + 1"),
        ],
        name="CycleDep",
        equations="d_x = -x * a + coupling.x",
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        build_model(spec)


def test_dependents_self_reference_detected():
    spec = _spec_with_params(
        [Parameter("a", formula="a + 1")],
        name="SelfDep",
        equations="d_x = -x * a + coupling.x",
    )
    with pytest.raises(ValueError, match="references itself"):
        build_model(spec)


def test_dependents_recompute_on_configure():
    """Modifying an independent param + re-configure() recomputes dependents."""
    spec = _spec_with_params(
        [
            Parameter("a", default=2.0),
            Parameter("b", formula="a * 10"),
        ],
        name="RecomputeDep",
        equations="d_x = -x / b + coupling.x",
    )
    Cls = build_model(spec)
    n = 3
    W = np.zeros((n, n))
    m = Cls(g=0.0).configure(weights=W)
    assert np.allclose(m.b, 20.0)

    m.a = 5.0
    m.configure(weights=W)
    assert np.allclose(m.b, 50.0)
    assert np.allclose(m.m[int(m.P.b)], 50.0)
