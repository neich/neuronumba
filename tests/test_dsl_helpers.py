"""Tests for `helpers=[...]` — user @nb.njit functions usable from DSL equations.

This is the v0.2 unlock for Zerlaut-class models that need custom subroutines
(`erfc_approx`, threshold functions) the DSL itself doesn't provide.
"""
import numpy as np
import numba as nb
import pytest

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


# A pair of @nb.njit helpers we'll reuse across tests.

@nb.njit(nb.f8[:](nb.f8[:]), cache=False)
def soft_clip(x):
    """Saturating nonlinearity, scalar→scalar in numba (vectorized over arr)."""
    return np.tanh(x)


@nb.njit(nb.f8[:](nb.f8[:], nb.f8[:]), cache=False)
def gated_sum(a, b):
    return a + 0.5 * b


def _spec_with_helpers(helpers, equations="d_x = -x + soft_clip(x) + coupling.x"):
    return ModelSpec(
        name="HelpersModel",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[Parameter("a", default=1.0)],
        equations=equations,
        helpers=helpers,
    )


def test_helper_callable_from_dfun():
    """The dfun should compile and produce sensible output when calling soft_clip."""
    Cls = build_model(_spec_with_helpers([soft_clip]))
    n = 4
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))

    f = m.get_numba_dfun()
    state = np.array([[0.5, 1.0, -0.5, 2.0]])
    coupling = np.zeros((1, n))
    ds, _ = f(state, coupling.copy())
    expected = -state[0] + np.tanh(state[0])
    np.testing.assert_allclose(ds[0], expected, atol=1e-12)


def test_helper_unused_is_silently_dropped():
    """Listing a helper that isn't referenced in equations is a no-op (not an error)."""
    Cls = build_model(_spec_with_helpers([soft_clip, gated_sum]))
    n = 3
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))
    f = m.get_numba_dfun()
    state = np.array([[0.0, 0.5, 1.0]])
    ds, _ = f(state, np.zeros((1, n)))
    np.testing.assert_allclose(ds[0], -state[0] + np.tanh(state[0]), atol=1e-12)


def test_two_helpers_referenced():
    """Multiple helpers in one equation."""
    spec = _spec_with_helpers(
        [soft_clip, gated_sum],
        equations="d_x = -x + gated_sum(x, soft_clip(x)) + coupling.x",
    )
    Cls = build_model(spec)
    n = 3
    m = Cls(g=0.0).configure(weights=np.zeros((n, n)))
    f = m.get_numba_dfun()
    state = np.array([[0.5, 1.0, -0.5]])
    ds, _ = f(state, np.zeros((1, n)))
    expected = -state[0] + state[0] + 0.5 * np.tanh(state[0])
    np.testing.assert_allclose(ds[0], expected, atol=1e-12)


def test_helper_unknown_name_in_equation_raises():
    """Calling a name that isn't a helper, param, or np func should still error."""
    with pytest.raises(ValueError, match="unknown identifier"):
        build_model(_spec_with_helpers(
            [soft_clip],
            equations="d_x = -x + not_a_helper(x) + coupling.x",
        ))


def test_helper_name_collision_with_param_rejected():
    @nb.njit
    def a(z):  # collides with parameter "a"
        return z

    with pytest.raises(ValueError, match="clash with"):
        build_model(_spec_with_helpers([a]))


def test_helper_name_collision_with_state_var_rejected():
    @nb.njit
    def x(z):  # collides with state var "x"
        return z

    with pytest.raises(ValueError, match="clash with"):
        build_model(_spec_with_helpers([x]))


def test_helper_name_collision_with_np_func_rejected():
    @nb.njit
    def exp(z):  # collides with whitelisted np.exp
        return z

    with pytest.raises(ValueError, match="clash with"):
        build_model(_spec_with_helpers([exp]))


def test_helper_duplicate_names_rejected():
    @nb.njit
    def helper(z):
        return z

    # Same helper twice — same __name__, same identity, but the validator
    # objects regardless because the source-generation can't distinguish.
    with pytest.raises(ValueError, match="must be unique"):
        build_model(_spec_with_helpers([helper, helper]))


def test_helper_lambda_rejected():
    """Lambdas have __name__ == '<lambda>'; we reject them upfront."""
    f = lambda z: z  # noqa: E731
    f.__name__ = ""  # simulate a missing/empty name
    with pytest.raises(ValueError, match="no __name__"):
        build_model(_spec_with_helpers([f]))


def test_helpers_not_listed_for_specs_without_them():
    """Specs with no helpers continue to work exactly as before."""
    Cls = build_model(_spec_with_helpers([], equations="d_x = -x * a + coupling.x"))
    m = Cls(g=0.0).configure(weights=np.zeros((4, 4)))
    f = m.get_numba_dfun()
    ds, _ = f(np.array([[0.5, 1.0, -0.5, 2.0]]), np.zeros((1, 4)))
    np.testing.assert_allclose(ds[0], -np.array([0.5, 1.0, -0.5, 2.0]), atol=1e-12)
