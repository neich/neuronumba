"""Regression test for the `super(type(self), self)` recursion trap in
`build_model`.

Background:
  The original prototype used `super(type(self), self)._init_dependant()` inside
  the closure attached to the generated class. This works only as long as no one
  subclasses the generated class — the moment a user does, `type(self)` becomes
  the user's subclass and `super(type(self), self)` re-resolves to the SAME
  generated method, causing infinite recursion.

  The fix captures the generated class in a closure cell so `super()` always
  walks past it, regardless of `type(self)`.
"""
import numpy as np

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


def test_subclass_overrides_init_dependant():
    """Direct subclass: override _init_dependant + super() must not recurse."""
    spec = ModelSpec(
        name="SubclassableDSL",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("a", default=2.0),
            Parameter("b", default=3.0),
            Parameter("c", formula="a + b"),
        ],
        equations="d_x = -x / c + coupling.x",
    )
    Base = build_model(spec)

    class Extended(Base):
        def _init_dependant(self):
            super()._init_dependant()
            self.user_extra = 42

    n = 4
    m = Extended(g=0.0).configure(weights=np.zeros((n, n)))
    assert np.allclose(m.c, 5.0)
    assert m.user_extra == 42


def test_grandchild_subclass_chain():
    """Two-level subclass: A(DSL) -> B(A). Each calls super(); none recurses."""
    spec = ModelSpec(
        name="GrandparentDSL",
        state_vars=[StateVar("x", initial=0.0)],
        coupling_vars=[CouplingVar("x", kind="linear")],
        observables=[],
        parameters=[
            Parameter("a", default=1.0),
            Parameter("d", formula="a * 10"),
        ],
        equations="d_x = -x * d + coupling.x",
    )
    Base = build_model(spec)

    class Mid(Base):
        def _init_dependant(self):
            super()._init_dependant()
            self.mid_marker = "mid"

    class Leaf(Mid):
        def _init_dependant(self):
            super()._init_dependant()
            self.leaf_marker = "leaf"

    m = Leaf(g=0.0).configure(weights=np.zeros((3, 3)))
    assert np.allclose(m.d, 10.0)
    assert m.mid_marker == "mid"
    assert m.leaf_marker == "leaf"
