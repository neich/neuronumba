"""Spec-level validation: Parameter() rules and dfun-equation rules."""
import pytest

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


# ----- Parameter validation rules (Parameter.__post_init__) ------------------

def test_param_default_and_formula_rejected():
    with pytest.raises(ValueError, match="both `default` and `formula`"):
        Parameter("x", default=1.0, formula="y + 1")


def test_param_required_and_formula_rejected():
    with pytest.raises(ValueError, match="`formula` and is `required`"):
        Parameter("x", required=True, formula="y + 1")


def test_param_neither_default_nor_required_nor_formula():
    with pytest.raises(ValueError, match="needs either `default`"):
        Parameter("x")


# ----- AST rewriter validation -----------------------------------------------

def _minimal_spec(equations, observables=()):
    return ModelSpec(
        name="MinModel",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=list(observables),
        parameters=[Parameter("a", default=1.0)],
        equations=equations,
    )


def test_dfun_unknown_identifier_raises():
    spec = _minimal_spec("d_x = -x * a + bogus_thing + coupling.x")
    with pytest.raises(ValueError, match="unknown identifier 'bogus_thing'"):
        build_model(spec)


def test_dfun_missing_derivative_raises():
    spec = _minimal_spec("foo = a * 2")
    with pytest.raises(ValueError, match="missing derivative"):
        build_model(spec)


def test_dfun_missing_observable_raises():
    spec = _minimal_spec(
        "d_x = -x * a + coupling.x",
        observables=["r"],
    )
    with pytest.raises(ValueError, match="declared observable.*never assigned"):
        build_model(spec)


# ----- Spec-level structural validation -------------------------------------

def test_coupling_var_must_be_state_var():
    spec = ModelSpec(
        name="BadCoupling",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("y", kind="linear")],  # 'y' not in state_vars
        observables=[],
        parameters=[Parameter("a", default=1.0)],
        equations="d_x = -x * a + coupling.y",
    )
    with pytest.raises(ValueError, match="not declared as a state var"):
        build_model(spec)
