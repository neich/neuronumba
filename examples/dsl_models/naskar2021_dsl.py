"""Naskar 2021 multiscale dynamic mean-field with inhibitory plasticity.

DSL equivalent of `neuronumba.simulator.models.naskar2021.Naskar2021` (104 LOC).
Three state vars (S_e, S_i, J — where J is the plasticity term that's a state
variable, not a parameter) and two observables (Ie, re).

Note: the hand-written sigmoid here has *no* eps-fix (unlike Deco2014). This
DSL spec faithfully reproduces that. `tests/test_dsl_equivalence.py::
test_equivalence_naskar2021` asserts dfun equivalence to 1e-10 — slightly
looser than Hopf/Deco because tiny denominator differences from float
reordering propagate without the eps-fix. The math is identical.
"""
from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


naskar_spec = ModelSpec(
    name="Naskar2021DSL",
    state_vars=[
        StateVar("S_e", initial=0.001, bounds=(0.0, 1.0)),
        StateVar("S_i", initial=0.001, bounds=(0.0, 1.0)),
        StateVar("J",   initial=1.0),
    ],
    coupling_vars=[CouplingVar("S_e", kind="linear")],
    observables=["Ie", "re"],
    parameters=[
        Parameter("t_glu",      default=7.46),    # glutamate concentration
        Parameter("t_gaba",     default=1.82),    # GABA concentration
        Parameter("We",         default=1.0),     # external→excitatory scaling
        Parameter("Wi",         default=0.7),     # external→inhibitory scaling
        Parameter("I0",         default=0.382),   # overall external input
        Parameter("w",          default=1.4),     # recurrent self-excitation weight
        Parameter("J_NMDA",     default=0.15),    # NMDA current
        Parameter("M_e",        default=1.0),
        Parameter("ae",         default=310.0),
        Parameter("be",         default=125.0),
        Parameter("de",         default=0.16),
        Parameter("ai",         default=615.0),
        Parameter("bi",         default=177.0),
        Parameter("di",         default=0.087),
        Parameter("M_i",        default=1.0),
        Parameter("alfa_e",     default=0.072),   # NMDA forward rate
        Parameter("alfa_i",     default=0.53),    # GABA forward rate
        Parameter("B_e",        default=0.0066),  # NMDA backward rate (ms^-1)
        Parameter("B_i",        default=0.18),    # GABA backward rate (ms^-1)
        Parameter("gamma",      default=1.0),     # plasticity learning rate
        Parameter("rho",        default=3.0),     # target firing rate (Hz)
        Parameter("I_external", default=0.0),
    ],
    # Divisions by 1000 convert to per-millisecond rates, matching the
    # hand-written model exactly.
    equations="""
        Ie = We * I0 + w * J_NMDA * S_e + J_NMDA * coupling.S_e - J * S_i + I_external
        Ii = Wi * I0 + J_NMDA * S_e - S_i

        y_e = M_e * (ae * Ie - be)
        re = y_e / (1.0 - np.exp(-de * y_e))

        y_i = M_i * (ai * Ii - bi)
        ri = y_i / (1.0 - np.exp(-di * y_i))

        d_S_e = -S_e * B_e + alfa_e * t_glu * (1.0 - S_e) * re / 1000.0
        d_S_i = -S_i * B_i + alfa_i * t_gaba * (1.0 - S_i) * ri / 1000.0
        d_J   = gamma * (ri / 1000.0) * (re - rho) / 1000.0
    """,
)

Naskar2021DSL = build_model(naskar_spec)
