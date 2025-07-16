import math
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import numba as nb
from scipy.optimize import fsolve
import scipy.special as sp_spec
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.fitting.fic.fic import FICHerzog2022
from neuronumba.numba_tools.addr import address_as_void_pointer
from neuronumba.numba_tools.functions import erfc_approx, erfc_complex_array
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model
from neuronumba.simulator.models import LinearCouplingModel

# Precompute constants for better performance
sqrt2 = math.sqrt(2.0)
sqrt2_inv = 1.0 / sqrt2

# Numba optimization flags
NUMBA_CACHE = True
NUMBA_FASTMATH = True
NUMBA_NOGIL = True

@nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W,
                          Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L,
                          N_tot, p_connect_e, p_connect_i, g, K_ext_e, K_ext_i):
    """
    Compute the mean characteristic of neurons.
    Inspired from the next repository :
    https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
    :param Fe: firing rate of excitatory population
    :param Fi: firing rate of inhibitory population
    :param Fe_ext: external excitatory input
    :param Fi_ext: external inhibitory input
    :param W: level of adaptation
    :param Q_e: excitatory quantal conductance
    :param tau_e: excitatory decay
    :param E_e: excitatory reversal potential
    :param Q_i: inhibitory quantal conductance
    :param tau_i: inhibitory decay
    :param E_i: inhibitory reversal potential
    :param E_L: leakage reversal voltage of neurons
    :param g_L: leak conductance
    :param C_m: membrane capacitance
    :param E_L: leak reversal potential
    :param N_tot: cell number
    :param p_connect_e: connectivity probability of excitatory neurons
    :param p_connect_i: connectivity probability of inhibitory neurons
    :param g: fraction of inhibitory cells
    :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
    """
    # firing rate
    # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
    fe = (Fe + 1.0e-6) * (1. - g) * p_connect_e * N_tot + Fe_ext * K_ext_e
    fi = (Fi + 1.0e-6) * g * p_connect_i * N_tot + Fi_ext * K_ext_i

    # conductance fluctuation and effective membrane time constant
    mu_Ge, mu_Gi = Q_e * tau_e * fe, Q_i * tau_i * fi  # Eqns 5 from [MV_2018]
    mu_G = g_L + mu_Ge + mu_Gi  # Eqns 6 from [MV_2018]
    T_m = C_m / mu_G  # Eqns 6 from [MV_2018]

    # membrane potential
    mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G  # Eqns 7 from [MV_2018]
    # post-synaptic membrane potential event s around muV
    U_e, U_i = Q_e / mu_G * (E_e - mu_V), Q_i / mu_G * (E_i - mu_V)
    # Standard deviation of the fluctuations
    # Eqns 8 from [MV_2018]
    sigma = fe * (U_e * tau_e) ** 2 / (2. * (tau_e + T_m)) + fi * (U_i * tau_i) ** 2 / (2. * (tau_i + T_m))
    sigma_V = np.where(sigma >= 0.0, np.sqrt(sigma), 1e-6)  # avoid numerical error but not use with numba
    # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
    T_V_numerator = (fe * (U_e * tau_e) ** 2 + fi * (U_i * tau_i) ** 2)
    T_V_denominator = (fe * (U_e * tau_e) ** 2 / (tau_e + T_m) + fi * (U_i * tau_i) ** 2 / (tau_i + T_m))
    # T_V = np.divide(T_V_numerator, T_V_denominator, out=np.ones_like(T_V_numerator),
    #                    where=T_V_denominator != 0.0) # avoid numerical error but not use with numba
    T_V = T_V_numerator / T_V_denominator
    return mu_V, sigma_V, T_V

@nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def threshold_func(muV, sigmaV, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
    """
    The threshold function of the neurons
    :param muV: mean of membrane voltage
    :param sigmaV: variance of membrane voltage
    :param TvN: autocorrelation time constant
    :param P: Fitted coefficients of the transfer functions
    :return: threshold of neurons
    """
    # Normalization factors page 48 after the equation 4 from [ZD_2018]
    muV0, DmuV0 = -60.0, 10.0
    sV0, DsV0 = 4.0, 6.0
    TvN0, DTvN0 = 0.5, 1.
    V = (muV - muV0) / DmuV0
    S = (sigmaV - sV0) / DsV0
    T = (TvN - TvN0) / DTvN0
    # Eqns 11 from [MV_2018]
    return P0 + P1 * V + P2 * S + P3 * T + P4 * V ** 2 + P5 * S ** 2 + P6 * T ** 2 + P7 * V * S + P8 * V * T + P9 * S * T

# @nb.njit(cache=True)
# def estimate_firing_rate(muV, sigmaV, Tv, Vthre):
#     """
#     The threshold function of the neurons
#     :param muV: mean of membrane voltage
#     :param sigmaV: variance of membrane voltage
#     :param Tv: autocorrelation time constant
#     :param Vthre:threshold of neurons
#     """
#     # Eqns 10 from [MV_2018]
#     return sp_spec.erfc((Vthre - muV) / (np.sqrt(2) * sigmaV)) / (2 * Tv)


@nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
def TF(fe, fi, fe_ext, fi_ext, W, Pv, E_L,
        Q_e, tau_e, E_e, Q_i, tau_i, E_i,
        g_L, C_m, N_tot, p_connect_e, p_connect_i, g, K_ext_e, K_ext_i):
    """
    transfer function for inhibitory population
    Inspired from the next repository :
    https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
    :param fe: firing rate of excitatory population
    :param fi: firing rate of inhibitory population
    :param fe_ext: external excitatory input
    :param fi_ext: external inhibitory input
    :param W: level of adaptation
    :param P: Polynome of neurons phenomenological threshold (order 9)
    :param E_L: leak reversal potential
    :return: result of transfer function
    """
    mu_V, sigma_V, T_V = get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, Q_e, tau_e, E_e,
                                                    Q_i, tau_i, E_i,
                                                    g_L, C_m, E_L, N_tot,
                                                    p_connect_e, p_connect_i, g, K_ext_e, K_ext_i)
    V_thre = threshold_func(mu_V, sigma_V, T_V * g_L / C_m,
                                Pv[0], Pv[1], Pv[2], Pv[3], Pv[4], Pv[5], Pv[6], Pv[7], Pv[8], Pv[9])
    V_thre *= 1e3  # the threshold need to be in mv and not in Volt
    f_out = erfc_approx((V_thre - mu_V) * sqrt2_inv / sigma_V) * 0.5 / T_V
    return f_out



class ZerlautAdaptationFirstOrder(LinearCouplingModel):
    r"""
    **References**:
    .. [ZD_2018]  Zerlaut, Y., Chemla, S., Chavane, F. et al. *Modeling mesoscopic cortical dynamics using a mean-field
    model of conductance-based networks of adaptive
    exponential integrate-and-fire neurons*,
    J Comput Neurosci (2018) 44: 45. https://doi-org.lama.univ-amu.fr/10.1007/s10827-017-0668-2
    .. [MV_2018]  Matteo di Volo, Alberto Romagnoni, Cristiano Capone, Alain Destexhe (2018)
    *Mean-field model for the dynamics of conductance-based networks of excitatory and inhibitory spiking neurons
    with adaptation*, bioRxiv, doi: https://doi.org/10.1101/352393

    Used Eqns 4 from [MV_2018]_ in ``dfun``.

    The default parameters are taken from table 1 of [ZD_2018]_, pag.47 and modify for the adaptation [MV_2018]
    +---------------------------+------------+
    |                 Table 1                |
    +--------------+------------+------------+
    |Parameter     |  Value     | Unit       |
    +==============+============+============+
    |             cellular property          |
    +--------------+------------+------------+
    | g_L          |   10.00    |   nS       |
    +--------------+------------+------------+
    | E_L_e        |  -60.00    |   mV       |
    +--------------+------------+------------+
    | E_L_i        |  -65.00    |   mV       |
    +--------------+------------+------------+
    | C_m          |   200.0    |   pF       |
    +--------------+------------+------------+
    | b_e          |   60.0     |   nS       |
    +--------------+------------+------------+
    | b_i          |   0.0      |   nS       |
    +--------------+------------+------------+
    | a_e          |   4.0      |   nS       |
    +--------------+------------+------------+
    | a_i          |   0.0      |   nS       |
    +--------------+------------+------------+
    | tau_w_e      |   500.0    |   ms       |
    +--------------+------------+------------+
    | tau_w_i      |   0.0      |   ms       |
    +--------------+------------+------------+
    | T            |   20.0      |   ms       |
    +--------------+------------+------------+
    |          synaptic properties           |
    +--------------+------------+------------+
    | E_e          |    0.0     | mV         |
    +--------------+------------+------------+
    | E_i          |   -80.0    | mV         |
    +--------------+------------+------------+
    | Q_e          |    1.0     | nS         |
    +--------------+------------+------------+
    | Q_i          |    5.0     | nS         |
    +--------------+------------+------------+
    | tau_e        |    5.0     | ms         |
    +--------------+------------+------------+
    | tau_i        |    5.0     | ms         |
    +--------------+------------+------------+
    |          numerical network             |
    +--------------+------------+------------+
    | N_tot        |  10000     |            |
    +--------------+------------+------------+
    | p_connect    |    5.0 %   |            |
    +--------------+------------+------------+
    | g            |   20.0 %   |            |
    +--------------+------------+------------+
    | K_e_ext      |   400      |            |
    +--------------+------------+------------+
    | K_i_ext      |   0        |            |
    +--------------+------------+------------+
    |external_input|    0.000   | Hz         |
    +--------------+------------+------------+

    The default coefficients of the transfer function are taken from table I of [MV_2018]_, pag.49
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |      excitatory cell      |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |  -4.98e-02  |   5.06e-03  |  -2.5e-02   |   1.4e-03   |  -4.1e-04   |   1.05e-02  |  -3.6e-02   |   7.4e-03   |   1.2e-03   |  -4.07e-02  |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |      inhibitory cell      |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
    |  -5.14e-02  |   4.0e-03   |  -8.3e-03   |   2.0e-04   |  -5.0e-04   |   1.4e-03   |  -1.46e-02  |   4.5e-03   |   2.8e-03   |  -1.53e-02  |
    +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+

    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its null-clines, using default parameters, can be
    seen below:

    .. automethod:: Zerlaut_adaptation_first_order.__init__

    The general formulation for the Zerlaut adaptation first order model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
        T\dot{E}_k &= F_e-E_k \\
        T\dot{I}_k &= F_i-I_k \\
        \dot{W}_k &= W_k/tau_w-b*E_k \\
        F_\lambda = Erfc(V^{eff}_{thre}-\mu_V/\sqrt(2)\sigma_V)

    """

    # E: firing rate of excitatory population in KHz\n
    # I: firing rate of inhibitory population in KHz\n
    # W_e: level of adaptation of excitatory in pA\n
    # W_i: level of adaptation of inhibitory in pA\n
    state_vars = Model._build_var_dict('E I W_e W_i ou_drift'.split())
    n_state_vars = len(state_vars)
    c_vars = [0]  # Only E couples between regions

    n_observable_vars = 0



    # Define traited attributes for this model, these represent possible kwargs.
    g_L = Attr(default=10.0, attributes=Model.Type.Model, doc="leak conductance [nS]")

    E_L_e = Attr(default=-65.0, attributes=Model.Type.Model, doc="leak reversal potential for excitatory [mV]")

    E_L_i = Attr(default=-65.0, attributes=Model.Type.Model, doc="leak reversal potential for inhibitory [mV]")

    # N.B. Not independent of g_L, C_m should scale linearly with g_L
    C_m = Attr(default=200.0, attributes=Model.Type.Model, doc="membrane capacitance [pF]")

    b_e = Attr(default=60.0, attributes=Model.Type.Model, doc="Excitatory adaptation current increment [pA]")

    a_e = Attr(default=4.0, attributes=Model.Type.Model, doc="Excitatory adaptation conductance [nS]")

    b_i = Attr(default=0.0, attributes=Model.Type.Model, doc="Inhibitory adaptation current increment [pA]")

    a_i = Attr(default=0.0, attributes=Model.Type.Model, doc="Inhibitory adaptation conductance [nS]")

    tau_w_e = Attr(default=500.0, attributes=Model.Type.Model, doc="Adaptation time constant of excitatory neurons [ms]")
    
    tau_w_i = Attr(default=1.0, attributes=Model.Type.Model, doc="Adaptation time constant of inhibitory neurons [ms]")

    E_e = Attr(default=0.0, attributes=Model.Type.Model, doc="excitatory reversal potential [mV]")

    E_i = Attr(default=-80.0, attributes=Model.Type.Model, doc="inhibitory reversal potential [mV]")

    Q_e = Attr(default=1.5, attributes=Model.Type.Model, doc="excitatory quantal conductance [nS]")

    Q_i = Attr(default=5.0, attributes=Model.Type.Model, doc="inhibitory quantal conductance [nS]")

    tau_e = Attr(default=5.0, attributes=Model.Type.Model, doc="excitatory decay [ms]")

    tau_i = Attr(default=5.0, attributes=Model.Type.Model, doc="inhibitory decay [ms]")

    N_tot = Attr(default=10000, attributes=Model.Type.Model, doc="cell number")

    p_connect_e = Attr(default=0.05, attributes=Model.Type.Model, doc="connectivity probability")

    p_connect_i = Attr(default=0.05, attributes=Model.Type.Model, doc="connectivity probability")

    gi = Attr(default=0.2, attributes=Model.Type.Model, doc="fraction of inhibitory cells")

    K_ext_e = Attr(default=400, attributes=Model.Type.Model, doc="Number of excitatory connexions from external population")

    K_ext_i = Attr(default=0, attributes=Model.Type.Model, doc="Number of inhibitory connexions from external population")

    T = Attr(default=20.0, attributes=Model.Type.Model, doc="time scale of describing network activity")

    P_e = Attr(default=np.array([-0.04983106, 0.005063550882777035, -0.023470121807314552,
                             0.0022951513725067503,
                             -0.0004105302652029825, 0.010547051343547399, -0.03659252821136933,
                             0.007437487505797858, 0.001265064721846073, -0.04072161294490446]),
                attributes=Model.Type.ModelAux,
                doc="Polynome of excitatory phenomenological threshold (order 9)")

    P_i = Attr(default=np.array([-0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
                             0.0002414237992765705,
                             -0.0005070645080016026, 0.0014345394104282397, -0.014686689498949967,
                             0.004502706285435741,
                             0.0028472190352532454, -0.015357804594594548]),
                attributes=Model.Type.ModelAux,
                doc="Polynome of inhibitory phenomenological threshold (order 9)")

    external_input_ex_ex = Attr(default=0.0, attributes=Model.Type.Model, doc="external drive")

    external_input_ex_in = Attr(default=0.0, attributes=Model.Type.Model, doc="external drive")

    external_input_in_ex = Attr(default=0.0, attributes=Model.Type.Model, doc="external drive")

    external_input_in_in = Attr(default=0.0, attributes=Model.Type.Model, doc="external drive")

    tau_OU = Attr(default=5.0, attributes=Model.Type.Model, doc="time constant noise")

    weight_noise = Attr(default=10.5, attributes=Model.Type.Model, doc="weight noise")

    S_i = Attr(default=1.0, attributes=Model.Type.Model, doc="Scaling of the remote input for the inhibitory population with respect to the excitatory population")

    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Initialize state variables for the model.
        
        Args:
            n_rois: Number of regions of interest
            
        Returns:
            Initial state array with shape (n_state_vars, n_rois)
        """
        state = np.empty((ZerlautAdaptationFirstOrder.n_state_vars, n_rois))
        state[:] = 0.001
        return state

    def get_numba_dfun(self):
        """
        Generate the Numba-compiled differential function for the Deco2014 model.
        
        Returns:
            Compiled function that computes state derivatives and observables
        """
        m = self.m.copy()
        m_aux = self.m_aux.copy()
        P = self.P
        P_aux = self.P_aux

        @nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def ZerlautAdaptationFirstOrder_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Numba-compiled function to compute the derivatives of the state variables
            and observables for the Zerlaut adaptation first order model.
            
            :param state: 2D array of state variables (shape: n_state_vars x n_nodes)
            :param coupling: 2D array of coupling values (shape: n_coupling_vars x n_nodes)
            :return: 2D array of derivatives (shape: n_state_vars x n_nodes)
            """
            # Precompute parameter values to avoid repeated array access
            T_val = m[np.intp(P.T)]
            inv_T = 1.0 / T_val
            weight_noise_val = m[np.intp(P.weight_noise)]
            K_ext_e_val = m[np.intp(P.K_ext_e)]
            
            # Precompute adaptation parameters
            tau_w_e_val = m[np.intp(P.tau_w_e)]
            tau_w_i_val = m[np.intp(P.tau_w_i)]
            b_e_val = m[np.intp(P.b_e)]
            b_i_val = m[np.intp(P.b_i)]
            a_e_val = m[np.intp(P.a_e)]
            a_i_val = m[np.intp(P.a_i)]
            E_L_e_val = m[np.intp(P.E_L_e)]
            E_L_i_val = m[np.intp(P.E_L_i)]
            
            # Precompute TF parameters
            P_e = m_aux[np.intp(P_aux.P_e)]
            P_i = m_aux[np.intp(P_aux.P_i)]
            Q_e_val = m[np.intp(P.Q_e)]
            tau_e_val = m[np.intp(P.tau_e)]
            E_e_val = m[np.intp(P.E_e)]
            Q_i_val = m[np.intp(P.Q_i)]
            tau_i_val = m[np.intp(P.tau_i)]
            E_i_val = m[np.intp(P.E_i)]
            g_L_val = m[np.intp(P.g_L)]
            C_m_val = m[np.intp(P.C_m)]
            N_tot_val = m[np.intp(P.N_tot)]
            p_connect_e_val = m[np.intp(P.p_connect_e)]
            p_connect_i_val = m[np.intp(P.p_connect_i)]
            g_val = m[np.intp(P.gi)]
            K_ext_i_val = m[np.intp(P.K_ext_i)]
            
            # External input parameters
            ext_ex_ex = m[np.intp(P.external_input_ex_ex)]
            ext_ex_in = m[np.intp(P.external_input_ex_in)]
            ext_in_ex = m[np.intp(P.external_input_in_ex)]
            ext_in_in = m[np.intp(P.external_input_in_in)]

            E = state[0, :]
            I = state[1, :]
            W_e = state[2, :]
            W_i = state[3, :]
            ou_drift = state[4, :]

            # long-range coupling
            c_0 = coupling[0, :]

            # external firing rate
            noise_term = weight_noise_val * ou_drift
            Fe_ext = c_0 + noise_term
            index_bad_input = np.where(Fe_ext * K_ext_e_val < 0)
            Fe_ext[index_bad_input] = 0.0

            Fi_ext = 0.0

            # Precompute external inputs
            Fe_ext_ex = Fe_ext + ext_ex_ex
            Fi_ext_ex = Fi_ext + ext_ex_in
            Fe_ext_in = Fe_ext + ext_in_ex
            Fi_ext_in = Fi_ext + ext_in_in

            # Excitatory firing rate derivation
            dE = (TF(
                E, I, Fe_ext_ex, Fi_ext_ex, W_e, P_e, E_L_e_val,
                Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                g_L_val, C_m_val, N_tot_val, p_connect_e_val, p_connect_i_val,
                g_val, K_ext_e_val, K_ext_i_val) - E) * inv_T
                
            # Inhibitory firing rate derivation
            dI = (TF(
                E, I, Fe_ext_in, Fi_ext_in, W_i, P_i, E_L_i_val,
                Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                g_L_val, C_m_val, N_tot_val, p_connect_e_val, p_connect_i_val,
                g_val, K_ext_e_val, K_ext_i_val) - I) * inv_T
                
            # Adaptation excitatory
            mu_V_e, sigma_V_e, T_V_e = get_fluct_regime_vars(
                E, I, Fe_ext_ex, Fi_ext_ex, W_e,
                Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                g_L_val, C_m_val, E_L_e_val, N_tot_val, p_connect_e_val,
                p_connect_i_val, g_val, K_ext_e_val, K_ext_i_val)
            dWe = -W_e / tau_w_e_val + b_e_val * E + a_e_val * (mu_V_e - E_L_e_val) / tau_w_e_val
            
            # Adaptation inhibitory
            mu_V_i, sigma_V_i, T_V_i = get_fluct_regime_vars(
                E, I, Fe_ext_in, Fi_ext_in, W_i,
                Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                g_L_val, C_m_val, E_L_i_val, N_tot_val, p_connect_e_val,
                p_connect_i_val, g_val, K_ext_e_val, K_ext_i_val)
            dWi = -W_i / tau_w_i_val + b_i_val * I + a_i_val * (mu_V_i - E_L_i_val) / tau_w_i_val
            
            dod = -ou_drift / m[np.intp(P.tau_OU)]

            return np.stack((dE, dI, dWe, dWi, dod)), np.empty((1,1))



class ZerlautAdaptationSecondOrder(ZerlautAdaptationFirstOrder):
    r"""
    **References**:
    .. [ZD_2018]  Zerlaut, Y., Chemla, S., Chavane, F. et al. *Modeling mesoscopic cortical dynamics using a mean-field
    model of conductance-based networks of adaptive
    exponential integrate-and-fire neurons*,
    J Comput Neurosci (2018) 44: 45. https://doi-org.lama.univ-amu.fr/10.1007/s10827-017-0668-2
    .. [MV_2018]  Matteo di Volo, Alberto Romagnoni, Cristiano Capone, Alain Destexhe (2018)
    *Mean-field model for the dynamics of conductance-based networks of excitatory and inhibitory spiking neurons
    with adaptation*, bioRxiv, doi: https://doi.org/10.1101/352393

    Used Eqns 4 from [MV_2018]_ in ``dfun``.

    (See Zerlaut_adaptation_first_order for the default value)

    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

    .. automethod:: Zerlaut_adaptation_second_order.__init__

    The general formulation for the \textit{\textbf{Zerlaut_adaptation_second_order}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
        \begin{aligned}
        \forall \mu,\lambda,\eta \in \{e,i\}^3\,
        &\left\{
        \begin{aligned}
        T \frac{\partial \nu_\mu}{\partial t} &= (\mathcal{F}_\mu - \nu_\mu )
        + \frac{1}{2} c_{\lambda \eta} 
        \frac{\partial^2 \mathcal{F}_\mu}{\partial \nu_\lambda \partial \nu_\eta} \\
        T \frac{\partial c_{\lambda \eta} }{\partial t}  &= A_{\lambda \eta} +
        (\mathcal{F}_\lambda - \nu_\lambda ) (\mathcal{F}_\eta - \nu_\eta )  \\
        &+ c_{\lambda \mu} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\lambda} +
        c_{\mu \eta} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\eta}
        - 2  c_{\lambda \eta}
        \end{aligned}\\
        \right. \\
        &\frac{\partial W_\mu}{\partial t} = \frac{W_\mu}{\tau_{w\mu}}-b_\mu*\nu_\mu +
        \frac{a_\mu(\mu_V-E_{L_\mu})}{\tau_{w\mu}}\\
        \end{aligned}

        with:
        A_{\lambda \eta} =
        \left\{
        \begin{split}
        \frac{\mathcal{F}_\lambda (1/T - \mathcal{F}_\lambda)}{N_\lambda}
        \qquad & \textrm{if  } \lambda=\eta \\
        0 \qquad & \textrm{otherwise}
        \end{split}
        \right.

        where:
        \begin{split}
        F_\lambda &= F_\lambda(\nu_e, \nu_i, \nu^{input\lambda}_e, \nu^{input\lambda}_i, W_\lambda)\\
        \nu^{input,e}_e &= \nu_{connectome} + \nu_{surface} + \nu^{ext}_ee + w_{noise}OU\\
        \nu^{input,e}_i &= S_i\nu_{connectome} + \nu_{surface} + \nu^{ext}_ei + w_{noise}OU\\
        \nu^{input,i}_e &= \nu_{surface} + \nu^{ext}_ie\\
        \nu^{input,i}_i &= \nu_{surface} + \nu^{ext}_ii\\
        \end{split}

        \textrm{given OU is the Ornstein-Uhlenbeck process}.
    """

    # The values for each state-variable should be set to encompass
    # the expected dynamic range of that state-variable for the current
    # parameters, it is used as a mechanism for bounding random inital
    # conditions when the simulation isn't started from an explicit history,
    # it is also provides the default range of phase-plane plots.\n
    # E: firing rate of excitatory population in KHz\n
    # I: firing rate of inhibitory population in KHz\n
    # C_ee: the variance of the excitatory population activity \n
    # C_ei: the covariance between the excitatory and inhibitory population activities (always symetric) \n
    # C_ie: the variance of the inhibitory population activity \n
    # W: level of adaptation
    state_vars = Model._build_var_dict('E I C_ee Cei C_ii W_e W_i ou_drift'.split())
    n_state_vars = len(state_vars)
    c_vars = [0]  # Only E couples between regions

    n_observable_vars = 0

    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Initialize state variables for the model.
        
        Args:
            n_rois: Number of regions of interest
            
        Returns:
            Initial state array with shape (n_state_vars, n_rois)
        """
        state = np.empty((ZerlautAdaptationSecondOrder.n_state_vars, n_rois))
        state[:] = 0.0
        return state

    def get_numba_dfun(self):
        """
        Generate the Numba-compiled differential function for the Deco2014 model.
        
        Returns:
            Compiled function that computes state derivatives and observables
        """
        m = self.m.copy()
        m_aux = self.m_aux.copy()
        P = self.P
        P_aux = self.P_aux
        P_e = m_aux[np.intp(P_aux.P_e)]
        P_i = m_aux[np.intp(P_aux.P_i)] 

        # Precompute parameter values to avoid repeated array access
        T_val = m[np.intp(P.T)]
        inv_T = 1.0 / T_val
        N_tot_val = m[np.intp(P.N_tot)]
        g_val = m[np.intp(P.gi)]
        S_i_val = m[np.intp(P.S_i)]
        weight_noise_val = m[np.intp(P.weight_noise)]

        # Neuron populations
        N_e = N_tot_val * (1 - g_val)
        N_i = N_tot_val * g_val

        # Precompute commonly used constants
        df = 1e-7
        df_1e3 = df * 1e3
        df_2_1e3 = 2 * df * 1e3
        df_1e3_sq = df_1e3 ** 2

        # Precompute all TF parameters once
        Q_e_val = m[np.intp(P.Q_e)]
        tau_e_val = m[np.intp(P.tau_e)]
        E_e_val = m[np.intp(P.E_e)]
        Q_i_val = m[np.intp(P.Q_i)]
        tau_i_val = m[np.intp(P.tau_i)]
        E_i_val = m[np.intp(P.E_i)]
        g_L_val = m[np.intp(P.g_L)]
        C_m_val = m[np.intp(P.C_m)]
        p_connect_e_val = m[np.intp(P.p_connect_e)]
        p_connect_i_val = m[np.intp(P.p_connect_i)]
        K_ext_e_val = m[np.intp(P.K_ext_e)]
        K_ext_i_val = m[np.intp(P.K_ext_i)]
        E_L_e_val = m[np.intp(P.E_L_e)]
        E_L_i_val = m[np.intp(P.E_L_i)]

        @nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def TF_excitatory(fe, fi, fe_ext, fi_ext, W):
            return TF(fe, fi, fe_ext, fi_ext, W,
                        P_e, E_L_e_val,
                        Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                        g_L_val, C_m_val, N_tot_val, p_connect_e_val, p_connect_i_val,
                        g_val, K_ext_e_val, K_ext_i_val)

        @nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def TF_inhibitory(fe, fi, fe_ext, fi_ext, W):
            return TF(fe, fi, fe_ext, fi_ext, W,
                    P_i, E_L_i_val,
                    Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                    g_L_val, C_m_val, N_tot_val, p_connect_e_val, p_connect_i_val,
                    g_val, K_ext_e_val, K_ext_i_val)



        @nb.njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def ZerlautAdaptationSecondOrder_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            r"""
            .. math::
                \begin{aligned}
                \forall \mu,\lambda,\eta \in \{e,i\}^3\,
                &\left\{
                \begin{aligned}
                T \frac{\partial \nu_\mu}{\partial t} &= (\mathcal{F}_\mu - \nu_\mu )
                + \frac{1}{2} c_{\lambda \eta} 
                \frac{\partial^2 \mathcal{F}_\mu}{\partial \nu_\lambda \partial \nu_\eta} \\
                T \frac{\partial c_{\lambda \eta} }{\partial t}  &= A_{\lambda \eta} +
                (\mathcal{F}_\lambda - \nu_\lambda ) (\mathcal{F}_\eta - \nu_\eta )  \\
                &+ c_{\lambda \mu} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\lambda} +
                c_{\mu \eta} \frac{\partial \mathcal{F}_\mu}{\partial \nu_\eta}
                - 2  c_{\lambda \eta}
                \end{aligned}\\
                \right. \\
                &\frac{\partial W_\mu}{\partial t} = \frac{W_\mu}{\tau_{w\mu}}-b_\mu*\nu_\mu +
                \frac{a_\mu(\mu_V-E_{L_\mu})}{\tau_{w\mu}}\\
                \end{aligned}

                with:
                A_{\lambda \eta} =
                \left\{
                \begin{split}
                \frac{\mathcal{F}_\lambda (1/T - \mathcal{F}_\lambda)}{N_\lambda}
                \qquad & \textrm{if  } \lambda=\eta \\
                0 \qquad & \textrm{otherwise}
                \end{split}
                \right.

                where:
                \begin{split}
                F_\lambda &= F_\lambda(\nu_e, \nu_i, \nu^{input\lambda}_e, \nu^{input\lambda}_i, W_\lambda)\\
                \nu^{input,e}_e &= \nu_{connectome} + \nu_{surface} + \nu^{ext}_ee + w_{noise}OU\\
                \nu^{input,e}_i &= S_i\nu_{connectome} + \nu_{surface} + \nu^{ext}_ei + w_{noise}OU\\
                \nu^{input,i}_e &= \nu_{surface} + \nu^{ext}_ie\\
                \nu^{input,i}_i &= \nu_{surface} + \nu^{ext}_ii\\
                \end{split}

                \textrm{given OU is the Ornstein-Uhlenbeck process}.
            """            
             
            # State variables
            E = state[0, :]
            I = state[1, :]
            C_ee = state[2, :]
            C_ei = state[3, :]
            C_ii = state[4, :]
            W_e = state[5, :]
            W_i = state[6, :]
            ou_drift = state[7,:]

            # Long-range coupling
            c_0 = coupling[0, :]

            # Short-range (local) coupling
            lc_E = 0.0 # local_coupling * E
            lc_I = 0.0 # local_coupling * I

            # External firing rate for the different populations
            noise_input = weight_noise_val * ou_drift
            
            E_input_excitatory = c_0 + lc_E + ext_ex_ex + noise_input
            index_bad_input = np.where(E_input_excitatory < 0)
            E_input_excitatory[index_bad_input] = 0.0

            E_input_inhibitory = S_i_val * c_0 + lc_E + ext_in_ex + noise_input
            index_bad_input = np.where(E_input_inhibitory < 0)
            E_input_inhibitory[index_bad_input] = 0.0

            I_input_excitatory = lc_I + ext_ex_in
            I_input_inhibitory = lc_I + ext_in_in

            # Compute all TF evaluations in one batch to improve cache efficiency
            # Base TF values
            _TF_e = TF_excitatory(E, I, E_input_excitatory, I_input_excitatory, W_e)
            _TF_i = TF_inhibitory(E, I, E_input_inhibitory, I_input_inhibitory, W_i)

            # Perturbed TF values for derivatives - compute all at once
            TF_e_plus_df = TF_excitatory(E + df, I, E_input_excitatory, I_input_excitatory, W_e)
            TF_e_minus_df = TF_excitatory(E - df, I, E_input_excitatory, I_input_excitatory, W_e)
            TF_e_I_plus_df = TF_excitatory(E, I + df, E_input_excitatory, I_input_excitatory, W_e)
            TF_e_I_minus_df = TF_excitatory(E, I - df, E_input_excitatory, I_input_excitatory, W_e)
            
            TF_i_E_plus_df = TF_inhibitory(E + df, I, E_input_inhibitory, I_input_inhibitory, W_i)
            TF_i_E_minus_df = TF_inhibitory(E - df, I, E_input_inhibitory, I_input_inhibitory, W_i)
            TF_i_plus_df = TF_inhibitory(E, I + df, E_input_inhibitory, I_input_inhibitory, W_i)
            TF_i_minus_df = TF_inhibitory(E, I - df, E_input_inhibitory, I_input_inhibitory, W_i)

            # Compute all derivatives at once using vectorized operations
            inv_df_2_1e3 = 1.0 / df_2_1e3
            inv_df_1e3_sq = 1.0 / df_1e3_sq
            
            # First derivatives
            diff_fe_TF_e = (TF_e_plus_df - TF_e_minus_df) * inv_df_2_1e3
            diff_fe_TF_i = (TF_i_E_plus_df - TF_i_E_minus_df) * inv_df_2_1e3
            diff_fi_TF_e = (TF_e_I_plus_df - TF_e_I_minus_df) * inv_df_2_1e3
            diff_fi_TF_i = (TF_i_plus_df - TF_i_minus_df) * inv_df_2_1e3

            # Second derivatives
            diff2_fe_fe_e = (TF_e_plus_df - 2.0 * _TF_e + TF_e_minus_df) * inv_df_1e3_sq
            diff2_fi_fi_e = (TF_e_I_plus_df - 2.0 * _TF_e + TF_e_I_minus_df) * inv_df_1e3_sq
            diff2_fe_fe_i = (TF_i_E_plus_df - 2.0 * _TF_i + TF_i_E_minus_df) * inv_df_1e3_sq
            diff2_fi_fi_i = (TF_i_plus_df - 2.0 * _TF_i + TF_i_minus_df) * inv_df_1e3_sq
            
            # Mixed derivatives - only compute if needed
            TF_e_both_plus = TF_excitatory(E + df, I + df, E_input_excitatory, I_input_excitatory, W_e)
            TF_e_both_minus = TF_excitatory(E - df, I - df, E_input_excitatory, I_input_excitatory, W_e)
            TF_i_both_plus = TF_inhibitory(E + df, I + df, E_input_inhibitory, I_input_inhibitory, W_i)
            TF_i_both_minus = TF_inhibitory(E - df, I - df, E_input_inhibitory, I_input_inhibitory, W_i)
            
            diff2_fe_fi_e = (TF_e_both_plus - TF_e_plus_df - TF_e_I_plus_df + 2.0*_TF_e - TF_e_minus_df - TF_e_I_minus_df + TF_e_both_minus) * (0.25 * inv_df_1e3_sq)
            diff2_fe_fi_i = (TF_i_both_plus - TF_i_E_plus_df - TF_i_plus_df + 2.0*_TF_i - TF_i_E_minus_df - TF_i_minus_df + TF_i_both_minus) * (0.25 * inv_df_1e3_sq)


            # Vectorized operations with parallel processing hints
            # Firing rate equations - use fused multiply-add operations
            half_inv_T = 0.5 * inv_T
            dE = (_TF_e - E + half_inv_T * (
                  C_ee * diff2_fe_fe_e + C_ei * diff2_fe_fi_e + C_ei * diff2_fe_fi_e + C_ii * diff2_fi_fi_e
                  )) * inv_T
                  
            dI = (_TF_i - I + half_inv_T * (
                  C_ee * diff2_fe_fe_i + C_ei * diff2_fe_fi_i + C_ei * diff2_fe_fi_i + C_ii * diff2_fi_fi_i
                  )) * inv_T
            # Covariance equations - optimized with precomputed values
            TF_e_diff = _TF_e - E
            TF_i_diff = _TF_i - I
            
            # Covariance excitatory-excitatory
            dCee = (_TF_e * (inv_T - _TF_e) / N_e + TF_e_diff * TF_e_diff +
                    2.0 * (C_ee * diff_fe_TF_e + C_ei * diff_fi_TF_e) - 2.0 * C_ee) * inv_T
                    
            # Covariance excitatory-inhibitory
            dCei = (TF_e_diff * TF_i_diff + 
                    C_ee * diff_fe_TF_e + C_ei * diff_fe_TF_i + C_ei * diff_fi_TF_e + C_ii * diff_fi_TF_i - 
                    2.0 * C_ei) * inv_T
                    
            # Covariance inhibitory-inhibitory
            dCii = (_TF_i * (inv_T - _TF_i) / N_i + TF_i_diff * TF_i_diff +
                    2.0 * (C_ii * diff_fi_TF_i + C_ei * diff_fe_TF_i) - 2.0 * C_ii) * inv_T

            # Precompute adaptation parameters once
            tau_w_e_val = m[np.intp(P.tau_w_e)]
            tau_w_i_val = m[np.intp(P.tau_w_i)]
            b_e_val = m[np.intp(P.b_e)]
            b_i_val = m[np.intp(P.b_i)]
            a_e_val = m[np.intp(P.a_e)]
            a_i_val = m[np.intp(P.a_i)]
            tau_OU_val = m[np.intp(P.tau_OU)]
            
            # Precompute external input parameters once
            ext_ex_ex = m[np.intp(P.external_input_ex_ex)]
            ext_in_ex = m[np.intp(P.external_input_in_ex)]
            ext_ex_in = m[np.intp(P.external_input_ex_in)]
            ext_in_in = m[np.intp(P.external_input_in_in)]

            # Adaptation excitatory - use precomputed values
            mu_V_e, sigma_V_e, T_V_e = get_fluct_regime_vars(
                E, I, E_input_excitatory, I_input_excitatory, W_e, 
                Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                g_L_val, C_m_val, E_L_e_val, N_tot_val, p_connect_e_val,
                p_connect_i_val, g_val, K_ext_e_val, K_ext_i_val)
            dWe = -W_e / tau_w_e_val + b_e_val * E + a_e_val * (mu_V_e - E_L_e_val) / tau_w_e_val

            # Adaptation inhibitory - use precomputed values
            mu_V_i, sigma_V_i, T_V_i = get_fluct_regime_vars(
                E, I, E_input_inhibitory, I_input_inhibitory, W_i, 
                Q_e_val, tau_e_val, E_e_val, Q_i_val, tau_i_val, E_i_val,
                g_L_val, C_m_val, E_L_i_val, N_tot_val, p_connect_e_val,
                p_connect_i_val, g_val, K_ext_e_val, K_ext_i_val)
            dWi = -W_i / tau_w_i_val + b_i_val * I + a_i_val * (mu_V_i - E_L_i_val) / tau_w_i_val

            dou = -ou_drift / tau_OU_val
            return np.stack((dE, dI, dCee, dCei, dCii, dWe, dWi, dou)), np.empty((1, 1))

        return ZerlautAdaptationSecondOrder_dfun
