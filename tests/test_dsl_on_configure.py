"""Tests for the `on_configure` imperative escape hatch.

Covers:
  - The callback fires on configure() and sees current weights/g.
  - Mutations land in self.m (the parameter matrix the dfun reads).
  - The callback runs AFTER dependent-parameter evaluation.
  - Re-configure re-runs the callback (matches the dependents protocol).
  - Real FIC use case: FICHerzog2022 computing J for a Deco-style spec.
"""
import numpy as np
import pytest

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


def _basic_spec(parameters, on_configure=None, equations="d_x = -x * a + coupling.x"):
    return ModelSpec(
        name="ConfModel",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=parameters,
        equations=equations,
        on_configure=on_configure,
    )


def test_on_configure_fires_with_self():
    captured = {}
    def hook(self):
        captured["weights_shape"] = self.weights.shape
        captured["g"] = self.g

    spec = _basic_spec([Parameter("a", default=1.0)], on_configure=hook)
    Cls = build_model(spec)
    n = 4
    Cls(g=0.7).configure(weights=np.eye(n))
    assert captured["weights_shape"] == (n, n)
    assert captured["g"] == 0.7


def test_on_configure_mutation_persists_on_instance():
    """Setting self.<param> in the hook persists on the instance and is
    visible to the dfun closure on the next get_numba_dfun() call."""
    def hook(self):
        # Compute J from weights/g — toy version of FIC.
        self.J = 2.0 * self.g * self.weights.sum(axis=0) + 1.0

    spec = _basic_spec(
        [Parameter("a", default=1.0), Parameter("J", default=0.0)],
        on_configure=hook,
    )
    Cls = build_model(spec)
    n = 5
    W = np.ones((n, n)) * 0.5
    np.fill_diagonal(W, 0)
    m = Cls(g=0.3).configure(weights=W)

    expected_J = 2.0 * 0.3 * W.sum(axis=0) + 1.0
    np.testing.assert_allclose(m.J, expected_J)
    # Confirm the dfun closure sees the hook-set value (build compiles cleanly).
    m.get_numba_dfun()


def test_on_configure_runs_after_dependents():
    """The hook can read computed dependent params (they're set first)."""
    captured = {}
    def hook(self):
        captured["b_at_hook_time"] = float(np.atleast_1d(self.b)[0])

    spec = _basic_spec(
        [
            Parameter("a", default=3.0),
            Parameter("b", formula="a * 100.0"),     # dependent
        ],
        on_configure=hook,
    )
    Cls = build_model(spec)
    Cls(g=0.0).configure(weights=np.zeros((2, 2)))
    assert captured["b_at_hook_time"] == 300.0


def test_on_configure_overrides_dependents():
    """Whatever the hook sets is final — it runs AFTER dependents."""
    def hook(self):
        self.b = 999.0

    spec = _basic_spec(
        [
            Parameter("a", default=1.0),
            Parameter("b", formula="a * 100.0"),
        ],
        on_configure=hook,
    )
    Cls = build_model(spec)
    m = Cls(g=0.0).configure(weights=np.zeros((2, 2)))
    np.testing.assert_allclose(m.b, 999.0)


def test_on_configure_reruns_on_reconfigure():
    """Same protocol as dependent params: every configure() re-runs the hook."""
    counter = {"n": 0}
    def hook(self):
        counter["n"] += 1
        self.J = float(counter["n"]) * 10.0

    spec = _basic_spec(
        [Parameter("a", default=1.0), Parameter("J", default=0.0)],
        on_configure=hook,
    )
    Cls = build_model(spec)
    W = np.zeros((3, 3))
    m = Cls(g=0.0).configure(weights=W)
    assert counter["n"] == 1
    np.testing.assert_allclose(m.J, 10.0)

    m.configure(weights=W)
    assert counter["n"] == 2
    np.testing.assert_allclose(m.J, 20.0)


def test_on_configure_with_fic_herzog():
    """Canonical use case: imperative FIC computation that doesn't fit a formula.

    `FICHerzog2022.compute_J(sc, g)` returns a per-region J. We stash the
    result on the model and verify the dfun sees it via self.m.
    """
    from neuronumba.fitting.fic.fic import FICHerzog2022

    def auto_fic(self):
        self.J = FICHerzog2022().compute_J(self.weights, self.g)

    spec = ModelSpec(
        name="MiniDeco",
        state_vars=[StateVar("S_e", initial=0.001, bounds=(0.0, 1.0))],
        coupling_vars=[CouplingVar("S_e", kind="linear")],
        observables=[],
        parameters=[
            Parameter("tau_e", default=100.0),
            Parameter("J",     default=0.0),  # placeholder; FIC overwrites
        ],
        equations="d_S_e = -S_e / tau_e + J * coupling.S_e",
        on_configure=auto_fic,
    )
    Cls = build_model(spec)

    rng = np.random.default_rng(0)
    W = rng.random((6, 6)); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    m = Cls(g=0.4).configure(weights=W)

    expected_J = 0.75 * 0.4 * W.sum(axis=0) + 1.0  # FICHerzog2022 closed form
    np.testing.assert_allclose(m.J, expected_J)
    # Confirm the dfun closure sees the hook-set value (build compiles cleanly).
    m.get_numba_dfun()


def test_on_configure_none_keeps_old_behavior():
    """When the hook is unset, configure() behaves exactly as before."""
    spec = _basic_spec([Parameter("a", default=2.0)])  # on_configure=None
    Cls = build_model(spec)
    m = Cls(g=0.5).configure(weights=np.eye(3))
    np.testing.assert_allclose(m.a, 2.0)
