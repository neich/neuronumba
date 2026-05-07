"""Tests for `ModelBuilder` — incremental, fluent construction of DSL models."""
import numpy as np
import numba as nb
import pytest

from neuronumba.simulator.models.dsl import (
    ModelBuilder, ModelSpec, build_model,
)


def test_chainable_build_matches_modelspec():
    """A chained builder produces the same compiled model as a literal ModelSpec."""
    Cls_b = (ModelBuilder("Hopf")
        .add_state("x", initial=0.0)
        .add_state("y", initial=0.0)
        .add_coupling("x", kind="diffusive")
        .add_param("a", default=-0.5)
        .add_param("omega", default=0.3)
        .add_equation("d_x = (a - x*x - y*y)*x - omega*y + coupling.x")
        .add_equation("d_y = (a - x*x - y*y)*y + omega*x")
        .build())

    n = 4
    W = np.zeros((n, n))
    m_b = Cls_b(g=0.0).configure(weights=W)
    f_b = m_b.get_numba_dfun()

    state = np.array([[0.1, 0.2, -0.3, 0.4],
                      [0.5, -0.1, 0.2, 0.0]])
    coup = np.zeros((1, n))
    ds_b, _ = f_b(state, coup)

    # Hand-derived expected derivatives.
    a, omega = -0.5, 0.3
    r2 = state[0]**2 + state[1]**2
    expected_dx = (a - r2) * state[0] - omega * state[1]
    expected_dy = (a - r2) * state[1] + omega * state[0]
    np.testing.assert_allclose(ds_b[0], expected_dx, atol=1e-12)
    np.testing.assert_allclose(ds_b[1], expected_dy, atol=1e-12)


def test_imperative_style_works():
    """Same builder used imperatively (no chaining) produces an equivalent class."""
    b = ModelBuilder("Imp")
    b.add_state("x", initial=0.0)
    b.add_coupling("x", kind="linear")
    b.add_param("k", default=2.0)
    b.add_equation("d_x = -k * x + coupling.x")
    Cls = b.build()

    m = Cls(g=0.0).configure(weights=np.zeros((3, 3)))
    f = m.get_numba_dfun()
    state = np.array([[1.0, 2.0, 3.0]])
    ds, _ = f(state, np.zeros((1, 3)))
    np.testing.assert_allclose(ds[0], -2.0 * state[0], atol=1e-12)


def test_spec_returns_modelspec_without_compiling():
    """`.spec()` should return a ModelSpec without going through build_model."""
    spec = (ModelBuilder("S")
        .add_state("x")
        .add_coupling("x")
        .add_param("a", default=1.0)
        .add_equation("d_x = -a*x + coupling.x")
        .spec())
    assert isinstance(spec, ModelSpec)
    assert spec.name == "S"
    assert len(spec.state_vars) == 1
    assert len(spec.parameters) == 1
    assert "d_x = -a*x + coupling.x" in spec.equations


def test_set_equations_replaces_block():
    """`.set_equations(...)` replaces any previously-added equation lines."""
    b = (ModelBuilder("S")
        .add_state("x")
        .add_coupling("x")
        .add_param("a", default=1.0)
        .add_equation("d_x = a*x")  # will be discarded
        .set_equations("d_x = -a*x + coupling.x"))
    spec = b.spec()
    assert spec.equations == "d_x = -a*x + coupling.x"


def test_helper_and_on_configure_wire_through():
    """`add_helper` and `on_configure` propagate to the spec."""
    @nb.njit(nb.f8[:](nb.f8[:]), cache=False)
    def squash(x):
        return np.tanh(x)

    captured = {"hit": False}
    def hook(self):
        captured["hit"] = True

    Cls = (ModelBuilder("WithExtras")
        .add_state("x", initial=0.0)
        .add_coupling("x")
        .add_param("k", default=1.5)
        .add_helper(squash)
        .on_configure(hook)
        .add_equation("d_x = squash(k * x) + coupling.x")
        .build())

    n = 3
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))
    assert captured["hit"] is True
    f = m.get_numba_dfun()
    state = np.array([[0.1, 0.2, 0.3]])
    ds, _ = f(state, np.zeros((1, n)))
    np.testing.assert_allclose(ds[0], np.tanh(1.5 * state[0]), atol=1e-12)


def test_add_observable_wires_through():
    """`add_observable` is propagated; an unassigned observable is rejected at build."""
    # Happy path: observable assigned in equations.
    Cls = (ModelBuilder("Obs")
        .add_state("x", initial=0.0)
        .add_coupling("x")
        .add_param("a", default=1.0)
        .add_observable("r")
        .add_equation("r = a * x")
        .add_equation("d_x = -r + coupling.x")
        .build())
    m = Cls(g=0.0).configure(weights=np.zeros((2, 2)))
    f = m.get_numba_dfun()
    state = np.array([[0.5, -0.5]])
    ds, obs = f(state, np.zeros((1, 2)))
    np.testing.assert_allclose(ds[0], -1.0 * state[0], atol=1e-12)
    np.testing.assert_allclose(obs[0], state[0], atol=1e-12)

    # Sad path: observable declared but never assigned -> build fails.
    bad = (ModelBuilder("ObsBad")
        .add_state("x", initial=0.0)
        .add_coupling("x")
        .add_param("a", default=1.0)
        .add_observable("r")
        .add_equation("d_x = -a * x + coupling.x"))
    with pytest.raises(ValueError, match="never assigned"):
        bad.build()


def test_dependent_param_via_builder():
    """Dependent params (`formula=...`) are added the same way as independents."""
    Cls = (ModelBuilder("Dep")
        .add_state("x")
        .add_coupling("x")
        .add_param("a", default=2.0)
        .add_param("b", default=3.0)
        .add_param("c", formula="a + b")
        .add_equation("d_x = -x / c + coupling.x")
        .build())
    m = Cls(g=0.0).configure(weights=np.zeros((2, 2)))
    np.testing.assert_allclose(m.c, 5.0)


def test_override_initial_propagates():
    Cls = (ModelBuilder("Init")
        .add_state("x", initial=0.0)
        .add_coupling("x")
        .add_param("a", default=1.0)
        .add_equation("d_x = -a*x + coupling.x")
        .override_initial("x", 0.7)
        .build())
    m = Cls(g=0.0).configure(weights=np.zeros((2, 2)))
    s0 = m.initial_state(n_rois=2)
    np.testing.assert_allclose(s0, np.full((1, 2), 0.7))


def test_duplicate_names_rejected():
    b = ModelBuilder("Dup").add_state("x")
    with pytest.raises(ValueError, match="Duplicate state"):
        b.add_state("x")

    b2 = ModelBuilder("Dup").add_param("a", default=1.0)
    with pytest.raises(ValueError, match="Duplicate parameter"):
        b2.add_param("a", default=2.0)

    b3 = ModelBuilder("Dup").add_coupling("x")
    with pytest.raises(ValueError, match="Duplicate coupling"):
        b3.add_coupling("x")


def test_empty_builder_rejected_at_spec():
    with pytest.raises(ValueError, match="no state variables"):
        ModelBuilder("Empty").spec()

    with pytest.raises(ValueError, match="no equations"):
        ModelBuilder("NoEq").add_state("x").spec()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ModelBuilder("")
    b = ModelBuilder("X").add_state("x").add_coupling("x").add_param("a", default=1.0)
    with pytest.raises(ValueError):
        b.add_equation("   ")
    with pytest.raises(TypeError):
        b.add_helper(42)
    with pytest.raises(TypeError):
        b.on_configure("not-callable")


def test_builder_equivalent_to_handwritten_modelspec():
    """A builder-built class should be byte-for-byte interchangeable with one
    constructed from a literal ModelSpec."""
    spec_literal = ModelSpec(
        name="Equiv",
        state_vars=[__import__("neuronumba.simulator.models.dsl", fromlist=["StateVar"]).StateVar("x", initial=0.0)],
        coupling_vars=[__import__("neuronumba.simulator.models.dsl", fromlist=["CouplingVar"]).CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[__import__("neuronumba.simulator.models.dsl", fromlist=["Parameter"]).Parameter("a", default=2.5)],
        equations="d_x = -a * x + coupling.x",
    )
    Cls_literal = build_model(spec_literal)

    Cls_builder = (ModelBuilder("Equiv")
        .add_state("x", initial=0.0)
        .add_coupling("x", kind="linear")
        .add_param("a", default=2.5)
        .add_equation("d_x = -a * x + coupling.x")
        .build())

    n = 4
    W = np.zeros((n, n))
    m_lit = Cls_literal(g=0.0).configure(weights=W)
    m_bld = Cls_builder(g=0.0).configure(weights=W)

    state = np.array([[0.5, -0.1, 0.3, 0.2]])
    coup = np.zeros((1, n))
    ds_lit, _ = m_lit.get_numba_dfun()(state, coup)
    ds_bld, _ = m_bld.get_numba_dfun()(state, coup)
    np.testing.assert_allclose(ds_lit, ds_bld, atol=0.0)
