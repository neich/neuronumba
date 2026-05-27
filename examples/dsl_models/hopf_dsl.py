"""Hopf supercritical bifurcation model, expressed in the neuronumba DSL.

This is the DSL equivalent of `neuronumba.simulator.models.hopf.Hopf` (152 LOC
hand-written). The math is identical — `tests/test_dsl_equivalence.py::
test_equivalence_hopf` compares the dfun and coupling-kernel outputs against
the hand-written class on random inputs and asserts agreement to 1e-12.

If this file changes, that test must still pass; if `hopf.py` changes (e.g. a
numerical bug fix), this file must follow.
"""
from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


hopf_spec = ModelSpec(
    name="HopfDSL",
    state_vars=[
        StateVar("x", initial=0.1),
        StateVar("y", initial=0.1),
    ],
    coupling_vars=[
        CouplingVar("x", kind="diffusive"),
        CouplingVar("y", kind="diffusive"),
    ],
    observables=[],
    parameters=[
        Parameter("a",          default=-0.5),
        Parameter("omega",      default=0.3),
        Parameter("I_external", default=0.0),
    ],
    equations="""
        d_x = (a - x*x - y*y) * x - omega * y + coupling.x + I_external
        d_y = (a - x*x - y*y) * y + omega * x + coupling.y
    """,
)

HopfDSL = build_model(hopf_spec)
