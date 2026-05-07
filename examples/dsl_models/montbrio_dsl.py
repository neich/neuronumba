"""Montbrio firing-rate model, expressed in the DSL.

DSL equivalent of `neuronumba.simulator.models.montbrio.Montbrio` (160 LOC).
Six state variables (r_e, r_i, u_e, u_i, S_ee, S_ie) and four *dependent*
parameters — values computed once at `configure()` time from the independent
parameters:

    J_N_ee = J_ee + g_ee * log(a_e)
    J_N_ie = J_ie + g_ie * log(a_e)
    J_G_ei = J_ei + g_ei * log(a_i)
    J_G_ii = J_ii + g_ii * log(a_i)

This is the canonical example of the DSL's dependent-parameter feature. Modify
any of `J_ee`, `g_ee`, `a_e`, `J_ie`, `g_ie`, `J_ei`, `g_ei`, `a_i`, `J_ii`, or
`g_ii` and call `configure()` again — the dependent values are re-derived
automatically.

`tests/test_dsl_equivalence.py::test_equivalence_montbrio` asserts dfun
equivalence to 1e-10 against the hand-written class.
"""
from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)


montbrio_spec = ModelSpec(
    name="MontbrioDSL",
    state_vars=[
        StateVar("r_e",  initial=0.0),
        StateVar("r_i",  initial=0.0),
        StateVar("u_e",  initial=0.0),
        StateVar("u_i",  initial=0.0),
        StateVar("S_ee", initial=0.0),
        StateVar("S_ie", initial=0.0),
    ],
    coupling_vars=[CouplingVar("S_ee", kind="linear")],
    observables=[],
    parameters=[
        # Time constants
        Parameter("tau_e",   default=10.0),
        Parameter("tau_i",   default=10.0),
        Parameter("tau_N",   default=10.0),
        # Firing-rate parameters
        Parameter("delta_e", default=1.0),
        Parameter("delta_i", default=1.0),
        Parameter("eta_e",   default=1.0),
        Parameter("eta_i",   default=1.0),
        # Synaptic parameters
        Parameter("a_e",     default=0.25),
        Parameter("a_i",     default=1.0),
        Parameter("g_e",     default=2.5),
        Parameter("g_i",     default=0.0),
        Parameter("g_ee",    default=2.5),
        Parameter("g_ei",    default=0.0),
        Parameter("g_ie",    default=2.5),
        Parameter("g_ii",    default=0.0),
        # External inputs
        Parameter("I_e_ext", default=0.0),
        Parameter("I_i_ext", default=0.0),
        # Coupling strengths
        Parameter("J_A",     default=1.0),
        Parameter("J_ee",    default=10.0),
        Parameter("J_ei",    default=10.0),
        Parameter("J_ie",    default=10.0),
        Parameter("J_ii",    default=10.0),
        Parameter("J",       default=10.0),
        # Dependent parameters — recomputed every configure()
        Parameter("J_N_ee",  formula="J_ee + g_ee * np.log(a_e)"),
        Parameter("J_N_ie",  formula="J_ie + g_ie * np.log(a_e)"),
        Parameter("J_G_ei",  formula="J_ei + g_ei * np.log(a_i)"),
        Parameter("J_G_ii",  formula="J_ii + g_ii * np.log(a_i)"),
    ],
    equations="""
        I_e = I_e_ext + tau_e * S_ee - J * J_G_ei * tau_i * r_i + J_A * tau_e * coupling.S_ee
        I_i = I_i_ext + tau_e * S_ie - J_G_ii * tau_i * r_i

        d_r_e = (delta_e / (np.pi * tau_e) + 2.0 * r_e * u_e - g_e * r_e) / tau_e
        d_r_i = (delta_i / (np.pi * tau_i) + 2.0 * r_i * u_i - g_i * r_i) / tau_i

        d_u_e = (eta_e + u_e * u_e - r_e * np.pi * tau_e * (r_e * np.pi * tau_e) + I_e) / tau_e
        d_u_i = (eta_i + u_i * u_i - r_i * np.pi * tau_i * (r_i * np.pi * tau_i) + I_i) / tau_i

        d_S_ee = (-S_ee + J_N_ee * r_e) / tau_N
        d_S_ie = (-S_ie + J_N_ie * r_e) / tau_N
    """,
)

MontbrioDSL = build_model(montbrio_spec)
