"""Built-in templates: ModelDef instances for existing neuronumba models.

These are hand-transcribed equations of the actual numba dfun bodies, in the
syntax expected by ``builder.py`` (state variables as ``S_e``, coupling inputs
as ``cpl_S_e``, parameters as plain identifiers, ``np`` available).
"""
from __future__ import annotations

from model_def import ModelDef, ObservableVarDef, ParamDef, StateVarDef


def blank() -> ModelDef:
    return ModelDef(
        name="MyModel",
        state_vars=[StateVarDef(name="x", initial=0.1, sigma=0.02, is_coupling=True)],
        params=[ParamDef(name="a", default=-1.0, tag="regional", doc="Local decay rate")],
        equations="dx = a * x + cpl_x\n",
    )


def deco2014() -> ModelDef:
    eqs = """\
# Excitatory and inhibitory currents (Deco et al. 2014, eqs. 5-6)
Ie = Jext_e * I0 + w * J_NMDA * S_e + J_NMDA * cpl_S_e - J * S_i + I_external
Ii = Jext_i * I0 + J_NMDA * S_e - S_i

# Firing rates via sigmoid transfer functions (eqs. 7-8)
y_e = ae * Ie - be
denom_e = 1.0 - np.exp(-de * y_e)
denom_e = np.where(np.abs(denom_e) < 1e-12, 1e-12, denom_e)
re = y_e / denom_e

y_i = ai * Ii - bi
denom_i = 1.0 - np.exp(-di * y_i)
denom_i = np.where(np.abs(denom_i) < 1e-12, 1e-12, denom_i)
ri = y_i / denom_i

# Synaptic gating derivatives (eqs. 9-10)
dS_e = -S_e / tau_e + gamma_e * (1.0 - S_e) * re
dS_i = -S_i / tau_i + gamma_i * ri
"""
    return ModelDef(
        name="Deco2014",
        state_vars=[
            StateVarDef(name="S_e", initial=0.001, lo=0.0, hi=1.0, sigma=1e-3, is_coupling=True),
            StateVarDef(name="S_i", initial=0.001, lo=0.0, hi=1.0, sigma=1e-3),
        ],
        observable_vars=[
            ObservableVarDef(name="Ie", doc="Excitatory current (nA)"),
            ObservableVarDef(name="re", doc="Excitatory firing rate (Hz)"),
        ],
        params=[
            ParamDef("tau_e", 100.0, "regional", "Excitatory time constant (ms)"),
            ParamDef("tau_i", 10.0, "regional", "Inhibitory time constant (ms)"),
            ParamDef("gamma_e", 0.000641, "regional", "Excitatory synaptic efficacy"),
            ParamDef("gamma_i", 0.001, "regional", "Inhibitory synaptic efficacy"),
            ParamDef("I0", 0.382, "regional", "Overall external input (nA)"),
            ParamDef("Jext_e", 1.0, "regional", "Excitatory external input scaling"),
            ParamDef("Jext_i", 0.7, "regional", "Inhibitory external input scaling"),
            ParamDef("w", 1.4, "regional", "Local recurrent excitation"),
            ParamDef("J_NMDA", 0.15, "regional", "NMDA coupling (nA)"),
            ParamDef("J", 1.0, "regional", "Local inhibitory coupling strength"),
            ParamDef("ae", 310.0, "regional", "Excitatory gain"),
            ParamDef("be", 125.0, "regional", "Excitatory threshold"),
            ParamDef("de", 0.16, "regional", "Excitatory slope"),
            ParamDef("ai", 615.0, "regional", "Inhibitory gain"),
            ParamDef("bi", 177.0, "regional", "Inhibitory threshold"),
            ParamDef("di", 0.087, "regional", "Inhibitory slope"),
            ParamDef("I_external", 0.0, "regional", "External stimulation (nA)"),
        ],
        equations=eqs,
    )


def hopf() -> ModelDef:
    eqs = """\
# Squared modulus of the complex state
r2 = x * x + y * y

# Cartesian normal form of the supercritical Hopf bifurcation (Deco et al. 2017).
# Note: cpl_x and cpl_y are the standard linear couplings g * (W.T @ state).
# The original Hopf model uses g * (W.T @ state - sum(W,2)*state) which differs
# by a diagonal term; for a faithful Hopf implementation, edit the exported
# Python module and override get_numba_coupling.
dx = (a - r2) * x - omega * y + cpl_x + I_external
dy = (a - r2) * y + omega * x + cpl_y
"""
    return ModelDef(
        name="Hopf",
        state_vars=[
            StateVarDef(name="x", initial=0.1, sigma=0.02, is_coupling=True),
            StateVarDef(name="y", initial=0.1, sigma=0.02, is_coupling=True),
        ],
        params=[
            ParamDef("a", -0.5, "regional", "Bifurcation parameter"),
            ParamDef("omega", 0.3, "regional", "Angular frequency"),
            ParamDef("I_external", 0.0, "regional", "External stimulation"),
        ],
        equations=eqs,
    )


def naskar2021() -> ModelDef:
    eqs = """\
# Multiscale DMF with inhibitory plasticity (Naskar et al. 2021).
Ie = We * I0 + w * J_NMDA * S_e + J_NMDA * cpl_S_e - J * S_i + I_external
Ii = Wi * I0 + J_NMDA * S_e - S_i

y_e = M_e * (ae * Ie - be)
denom_e = 1.0 - np.exp(-de * y_e)
denom_e = np.where(np.abs(denom_e) < 1e-12, 1e-12, denom_e)
re = y_e / denom_e

y_i = M_i * (ai * Ii - bi)
denom_i = 1.0 - np.exp(-di * y_i)
denom_i = np.where(np.abs(denom_i) < 1e-12, 1e-12, denom_i)
ri = y_i / denom_i

dS_e = -S_e * B_e + alfa_e * t_glu * (1.0 - S_e) * re / 1000.0
dS_i = -S_i * B_i + alfa_i * t_gaba * (1.0 - S_i) * ri / 1000.0
dJ   = gamma * (ri / 1000.0) * (re - rho) / 1000.0
"""
    return ModelDef(
        name="Naskar2021",
        state_vars=[
            StateVarDef(name="S_e", initial=0.001, lo=0.0, hi=1.0, sigma=1e-3, is_coupling=True),
            StateVarDef(name="S_i", initial=0.001, lo=0.0, hi=1.0, sigma=1e-3),
            StateVarDef(name="J",   initial=1.0,   sigma=0.0),
        ],
        observable_vars=[
            ObservableVarDef(name="Ie", doc="Excitatory current (nA)"),
            ObservableVarDef(name="re", doc="Excitatory firing rate (Hz)"),
        ],
        params=[
            ParamDef("t_glu", 7.46, "regional", "Glutamate concentration"),
            ParamDef("t_gaba", 1.82, "regional", "GABA concentration"),
            ParamDef("We", 1.0, "regional"),
            ParamDef("Wi", 0.7, "regional"),
            ParamDef("I0", 0.382, "regional", "External input (nA)"),
            ParamDef("w", 1.4, "regional", "Recurrent excitation"),
            ParamDef("J_NMDA", 0.15, "regional", "NMDA coupling (nA)"),
            ParamDef("M_e", 1.0, "regional"),
            ParamDef("ae", 310.0, "regional"),
            ParamDef("be", 125.0, "regional"),
            ParamDef("de", 0.16, "regional"),
            ParamDef("ai", 615.0, "regional"),
            ParamDef("bi", 177.0, "regional"),
            ParamDef("di", 0.087, "regional"),
            ParamDef("M_i", 1.0, "regional"),
            ParamDef("alfa_e", 0.072, "regional", "NMDA forward rate"),
            ParamDef("alfa_i", 0.53, "regional", "GABA forward rate"),
            ParamDef("B_e", 0.0066, "regional", "NMDA backward rate (1/ms)"),
            ParamDef("B_i", 0.18, "regional", "GABA backward rate (1/ms)"),
            ParamDef("gamma", 1.0, "regional", "Plasticity learning rate"),
            ParamDef("rho", 3.0, "regional", "Target excitatory firing rate (Hz)"),
            ParamDef("I_external", 0.0, "regional"),
        ],
        equations=eqs,
    )


TEMPLATES = {
    "Blank": blank,
    "Deco2014": deco2014,
    "Hopf (linear-coupling approx.)": hopf,
    "Naskar2021": naskar2021,
}
