"""Zerlaut adaptation models, expressed in the neuronumba DSL.

DSL equivalents of:
  - `neuronumba.simulator.models.zerlaut.ZerlautAdaptationFirstOrder`
  - `neuronumba.simulator.models.zerlaut.ZerlautAdaptationSecondOrder`

The math is identical. The four njit subroutines used inside the dfun
(`get_fluct_regime_vars`, `threshold_func`, `TF`, `erfc_approx`) are imported
verbatim from the hand-written module so both versions share the exact same
helper bytecode — only the dfun itself is regenerated from the spec.

Two patterns the DSL needs (and now provides):
  - Tuple unpacking from a multi-return helper:
        ``mu_V, sigma_V, T_V = get_fluct_regime_vars(...)``
  - Long argument lists (TF takes 21 positional args). The DSL just transcribes
    them; numba treats this exactly the same as the hand-written code.

The hand-written 1st- and 2nd-order dfuns clamp inputs via fancy-index
assignment after `np.where(... < 0)`. The DSL spec uses ``np.maximum(x, 0.0)``
which is bit-equivalent for the values these tensors actually take (K_ext_e,
K_ext_i are non-negative by construction).

`tests/test_dsl_equivalence.py::test_equivalence_zerlaut_*` compares the
DSL-built dfuns to the hand-written ones on random inputs at 1e-12.
"""
import numpy as np

from neuronumba.simulator.models.dsl import (
    ModelSpec, StateVar, CouplingVar, Parameter, build_model,
)
# Helpers come straight from the hand-written file: bit-identical bytecode.
from neuronumba.simulator.models.zerlaut import (
    get_fluct_regime_vars, threshold_func, TF,
)


# ---- shared parameter list --------------------------------------------------

_P_e_DEFAULT = np.array([
    -0.04983106, 0.005063550882777035, -0.023470121807314552,
    0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
    -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
    -0.04072161294490446,
])
_P_i_DEFAULT = np.array([
    -0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
    0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
    -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
    -0.015357804594594548,
])


def _common_parameters():
    """The 32 parameters shared by both 1st- and 2nd-order Zerlaut models."""
    return [
        # Cellular properties
        Parameter("g_L",       default=10.0),
        Parameter("E_L_e",     default=-65.0),
        Parameter("E_L_i",     default=-65.0),
        Parameter("C_m",       default=200.0),
        Parameter("b_e",       default=60.0),
        Parameter("a_e",       default=4.0),
        Parameter("b_i",       default=0.0),
        Parameter("a_i",       default=0.0),
        Parameter("tau_w_e",   default=500.0),
        Parameter("tau_w_i",   default=1.0),
        # Synaptic properties
        Parameter("E_e",       default=0.0),
        Parameter("E_i",       default=-80.0),
        Parameter("Q_e",       default=1.5),
        Parameter("Q_i",       default=5.0),
        Parameter("tau_e",     default=5.0),
        Parameter("tau_i",     default=5.0),
        # Network composition
        Parameter("N_tot",       default=10000.0),
        Parameter("p_connect_e", default=0.05),
        Parameter("p_connect_i", default=0.05),
        Parameter("gi",          default=0.2),
        Parameter("K_ext_e",     default=400.0),
        Parameter("K_ext_i",     default=0.0),
        # Time scale
        Parameter("T", default=20.0),
        # Polynomial coefficients (length-10 1D arrays)
        Parameter("P_e", default=_P_e_DEFAULT),
        Parameter("P_i", default=_P_i_DEFAULT),
        # External drives & noise
        Parameter("external_input_ex_ex", default=0.0),
        Parameter("external_input_ex_in", default=0.0),
        Parameter("external_input_in_ex", default=0.0),
        Parameter("external_input_in_in", default=0.0),
        Parameter("tau_OU",       default=5.0),
        Parameter("weight_noise", default=10.5),
        Parameter("S_i",          default=1.0),
        # Factory-level constant: 1/T precomputed at configure() time so the
        # dfun multiplies by inv_T (matches the hand-written code's hoisting
        # exactly and is bit-equivalent to it).
        Parameter("inv_T", formula="1.0 / T"),
    ]


# ---- First-order model ------------------------------------------------------

zerlaut_first_order_spec = ModelSpec(
    name="ZerlautAdaptationFirstOrderDSL",
    state_vars=[
        StateVar("E",        initial=0.001),
        StateVar("I",        initial=0.001),
        StateVar("W_e",      initial=0.001),
        StateVar("W_i",      initial=0.001),
        StateVar("ou_drift", initial=0.001),
    ],
    coupling_vars=[CouplingVar("E", kind="linear")],
    observables=[],
    parameters=_common_parameters(),
    helpers=[get_fluct_regime_vars, threshold_func, TF],
    equations="""
        # External firing rate (clamped to non-negative; bit-equivalent to the
        # hand-written `np.where(Fe_ext * K_ext_e < 0); Fe_ext[idx] = 0` since
        # K_ext_e is non-negative).
        Fe_ext_raw = coupling.E + weight_noise * ou_drift
        Fe_ext = np.maximum(Fe_ext_raw, 0.0)

        # Population-specific external inputs. Fi_ext = 0 in the hand-written
        # code, so Fi_ext_ex / Fi_ext_in collapse to the corresponding ext_*_in
        # parameters.
        Fe_ext_ex = Fe_ext + external_input_ex_ex
        Fi_ext_ex = external_input_ex_in
        Fe_ext_in = Fe_ext + external_input_in_ex
        Fi_ext_in = external_input_in_in

        # Excitatory firing rate: dE = (TF_e - E) / T
        TF_e = TF(E, I, Fe_ext_ex, Fi_ext_ex, W_e, P_e, E_L_e,
                  Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                  g_L, C_m, N_tot, p_connect_e, p_connect_i, gi,
                  K_ext_e, K_ext_i)
        d_E = (TF_e - E) * inv_T

        # Inhibitory firing rate: dI = (TF_i - I) / T
        TF_i = TF(E, I, Fe_ext_in, Fi_ext_in, W_i, P_i, E_L_i,
                  Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                  g_L, C_m, N_tot, p_connect_e, p_connect_i, gi,
                  K_ext_e, K_ext_i)
        d_I = (TF_i - I) * inv_T

        # Excitatory adaptation
        mu_V_e, sigma_V_e, T_V_e = get_fluct_regime_vars(
            E, I, Fe_ext_ex, Fi_ext_ex, W_e,
            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
            g_L, C_m, E_L_e, N_tot, p_connect_e, p_connect_i, gi,
            K_ext_e, K_ext_i)
        d_W_e = -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e

        # Inhibitory adaptation
        mu_V_i, sigma_V_i, T_V_i = get_fluct_regime_vars(
            E, I, Fe_ext_in, Fi_ext_in, W_i,
            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
            g_L, C_m, E_L_i, N_tot, p_connect_e, p_connect_i, gi,
            K_ext_e, K_ext_i)
        d_W_i = -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i

        # OU drift relaxation (noise injection happens in the integrator)
        d_ou_drift = -ou_drift / tau_OU
    """,
)

ZerlautAdaptationFirstOrderDSL = build_model(zerlaut_first_order_spec)


# ---- Second-order model -----------------------------------------------------

zerlaut_second_order_spec = ModelSpec(
    name="ZerlautAdaptationSecondOrderDSL",
    state_vars=[
        StateVar("E",        initial=0.0),
        StateVar("I",        initial=0.0),
        StateVar("C_ee",     initial=0.0),
        StateVar("C_ei",     initial=0.0),
        StateVar("C_ii",     initial=0.0),
        StateVar("W_e",      initial=0.0),
        StateVar("W_i",      initial=0.0),
        StateVar("ou_drift", initial=0.0),
    ],
    coupling_vars=[CouplingVar("E", kind="linear")],
    observables=[],
    parameters=_common_parameters() + [
        # 2nd-order-specific factory-level precomputes (mirror the hand-written
        # constants computed at the top of get_numba_dfun).
        Parameter("N_e", formula="N_tot * (1.0 - gi)"),
        Parameter("N_i", formula="N_tot * gi"),
    ],
    helpers=[get_fluct_regime_vars, threshold_func, TF],
    equations="""
        # Finite-difference step constants (factory-equivalent literals; numba
        # constant-folds these).
        df = 1e-7
        df_1e3 = df * 1e3
        df_2_1e3 = 2 * df * 1e3
        df_1e3_sq = df_1e3 * df_1e3
        inv_df_2_1e3 = 1.0 / df_2_1e3
        inv_df_1e3_sq = 1.0 / df_1e3_sq

        # External firing-rate inputs (clamped to non-negative).
        noise_input = weight_noise * ou_drift
        E_input_excitatory_raw = coupling.E + external_input_ex_ex + noise_input
        E_input_excitatory = np.maximum(E_input_excitatory_raw, 0.0)
        E_input_inhibitory_raw = S_i * coupling.E + external_input_in_ex + noise_input
        E_input_inhibitory = np.maximum(E_input_inhibitory_raw, 0.0)
        I_input_excitatory = external_input_ex_in
        I_input_inhibitory = external_input_in_in

        # Base TF values (one for each population).
        _TF_e = TF(E, I, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                   Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                   g_L, C_m, N_tot, p_connect_e, p_connect_i, gi,
                   K_ext_e, K_ext_i)
        _TF_i = TF(E, I, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                   Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                   g_L, C_m, N_tot, p_connect_e, p_connect_i, gi,
                   K_ext_e, K_ext_i)

        # Perturbed TF values for centered-FD derivatives.
        TF_e_plus_df = TF(E + df, I, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                          Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                          g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_e_minus_df = TF(E - df, I, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                           Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                           g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_e_I_plus_df = TF(E, I + df, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                            g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_e_I_minus_df = TF(E, I - df, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                             Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                             g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)

        TF_i_E_plus_df = TF(E + df, I, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                            g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_i_E_minus_df = TF(E - df, I, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                             Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                             g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_i_plus_df = TF(E, I + df, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                          Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                          g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_i_minus_df = TF(E, I - df, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                           Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                           g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)

        # First derivatives.
        diff_fe_TF_e = (TF_e_plus_df - TF_e_minus_df) * inv_df_2_1e3
        diff_fe_TF_i = (TF_i_E_plus_df - TF_i_E_minus_df) * inv_df_2_1e3
        diff_fi_TF_e = (TF_e_I_plus_df - TF_e_I_minus_df) * inv_df_2_1e3
        diff_fi_TF_i = (TF_i_plus_df - TF_i_minus_df) * inv_df_2_1e3

        # Second derivatives (single-variable).
        diff2_fe_fe_e = (TF_e_plus_df - 2.0 * _TF_e + TF_e_minus_df) * inv_df_1e3_sq
        diff2_fi_fi_e = (TF_e_I_plus_df - 2.0 * _TF_e + TF_e_I_minus_df) * inv_df_1e3_sq
        diff2_fe_fe_i = (TF_i_E_plus_df - 2.0 * _TF_i + TF_i_E_minus_df) * inv_df_1e3_sq
        diff2_fi_fi_i = (TF_i_plus_df - 2.0 * _TF_i + TF_i_minus_df) * inv_df_1e3_sq

        # Mixed derivatives (need both-corner perturbations).
        TF_e_both_plus = TF(E + df, I + df, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                            g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_e_both_minus = TF(E - df, I - df, E_input_excitatory, I_input_excitatory, W_e, P_e, E_L_e,
                             Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                             g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_i_both_plus = TF(E + df, I + df, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                            g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)
        TF_i_both_minus = TF(E - df, I - df, E_input_inhibitory, I_input_inhibitory, W_i, P_i, E_L_i,
                             Q_e, tau_e, E_e, Q_i, tau_i, E_i,
                             g_L, C_m, N_tot, p_connect_e, p_connect_i, gi, K_ext_e, K_ext_i)

        diff2_fe_fi_e = (TF_e_both_plus - TF_e_plus_df - TF_e_I_plus_df + 2.0*_TF_e - TF_e_minus_df - TF_e_I_minus_df + TF_e_both_minus) * (0.25 * inv_df_1e3_sq)
        diff2_fe_fi_i = (TF_i_both_plus - TF_i_E_plus_df - TF_i_plus_df + 2.0*_TF_i - TF_i_E_minus_df - TF_i_minus_df + TF_i_both_minus) * (0.25 * inv_df_1e3_sq)

        # Firing-rate equations.
        half_inv_T = 0.5 * inv_T
        d_E = (_TF_e - E + half_inv_T * (
              C_ee * diff2_fe_fe_e + C_ei * diff2_fe_fi_e + C_ei * diff2_fe_fi_e + C_ii * diff2_fi_fi_e
              )) * inv_T
        d_I = (_TF_i - I + half_inv_T * (
              C_ee * diff2_fe_fe_i + C_ei * diff2_fe_fi_i + C_ei * diff2_fe_fi_i + C_ii * diff2_fi_fi_i
              )) * inv_T

        # Covariance equations.
        TF_e_diff = _TF_e - E
        TF_i_diff = _TF_i - I
        d_C_ee = (_TF_e * (inv_T - _TF_e) / N_e + TF_e_diff * TF_e_diff +
                  2.0 * (C_ee * diff_fe_TF_e + C_ei * diff_fi_TF_e) - 2.0 * C_ee) * inv_T
        d_C_ei = (TF_e_diff * TF_i_diff +
                  C_ee * diff_fe_TF_e + C_ei * diff_fe_TF_i + C_ei * diff_fi_TF_e + C_ii * diff_fi_TF_i -
                  2.0 * C_ei) * inv_T
        d_C_ii = (_TF_i * (inv_T - _TF_i) / N_i + TF_i_diff * TF_i_diff +
                  2.0 * (C_ii * diff_fi_TF_i + C_ei * diff_fe_TF_i) - 2.0 * C_ii) * inv_T

        # Adaptation (excitatory).
        mu_V_e, sigma_V_e, T_V_e = get_fluct_regime_vars(
            E, I, E_input_excitatory, I_input_excitatory, W_e,
            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
            g_L, C_m, E_L_e, N_tot, p_connect_e, p_connect_i, gi,
            K_ext_e, K_ext_i)
        d_W_e = -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e

        # Adaptation (inhibitory).
        mu_V_i, sigma_V_i, T_V_i = get_fluct_regime_vars(
            E, I, E_input_inhibitory, I_input_inhibitory, W_i,
            Q_e, tau_e, E_e, Q_i, tau_i, E_i,
            g_L, C_m, E_L_i, N_tot, p_connect_e, p_connect_i, gi,
            K_ext_e, K_ext_i)
        d_W_i = -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i

        # OU drift relaxation.
        d_ou_drift = -ou_drift / tau_OU
    """,
)

ZerlautAdaptationSecondOrderDSL = build_model(zerlaut_second_order_spec)
