"""Reduced Wong-Wang / Deco 2014 dynamic mean-field, expressed in the DSL.

DSL equivalent of `neuronumba.simulator.models.deco2014.Deco2014` (416 LOC).
Note this spec covers only the dfun + linear coupling — the hand-written class
also has `auto_fic` (FIC steady-state computation) which is not part of the
core differential equations. To use FIC with this DSL model you would override
`_init_dependant` in a subclass. See `tests/test_dsl_subclass.py` for the
subclassing pattern.

`tests/test_dsl_equivalence.py::test_equivalence_deco2014` asserts dfun
equivalence against the hand-written class to 1e-12 (with `auto_fic=False` so
both share the same `J=1.0`).
"""
from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


# The hand-written Deco2014 has an EPS-fix to avoid div-by-0 in the firing-rate
# sigmoid. We faithfully reproduce that with `np.where`.
deco_spec = ModelSpec(
    name="Deco2014DSL",
    state_vars=[
        StateVar("S_e", initial=0.001, bounds=(0.0, 1.0)),
        StateVar("S_i", initial=0.001, bounds=(0.0, 1.0)),
    ],
    coupling_vars=[CouplingVar("S_e", kind="linear")],
    observables=["Ie", "re"],
    parameters=[
        Parameter("tau_e",      default=100.0),
        Parameter("tau_i",      default=10.0),
        Parameter("gamma_e",    default=0.000641),
        Parameter("gamma_i",    default=0.001),
        Parameter("I0",         default=0.382),
        Parameter("Jext_e",     default=1.0),
        Parameter("Jext_i",     default=0.7),
        Parameter("w",          default=1.4),
        Parameter("J_NMDA",     default=0.15),
        Parameter("J",          default=1.0),
        Parameter("ae",         default=310.0),
        Parameter("be",         default=125.0),
        Parameter("de",         default=0.16),
        Parameter("ai",         default=615.0),
        Parameter("bi",         default=177.0),
        Parameter("di",         default=0.087),
        Parameter("I_external", default=0.0),
    ],
    equations="""
        Ie = Jext_e * I0 + w * J_NMDA * S_e + J_NMDA * coupling.S_e - J * S_i + I_external
        Ii = Jext_i * I0 + J_NMDA * S_e - S_i

        y_e = ae * Ie - be
        denom_e = 1.0 - np.exp(-de * y_e)
        denom_e = np.where(np.abs(denom_e) < 1e-12, 1e-12, denom_e)
        re = y_e / denom_e

        y_i = ai * Ii - bi
        denom_i = 1.0 - np.exp(-di * y_i)
        denom_i = np.where(np.abs(denom_i) < 1e-12, 1e-12, denom_i)
        ri = y_i / denom_i

        d_S_e = -S_e / tau_e + gamma_e * (1.0 - S_e) * re
        d_S_i = -S_i / tau_i + gamma_i * ri
    """,
)

Deco2014DSL = build_model(deco_spec)
