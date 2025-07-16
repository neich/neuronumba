# ==========================================================================
# ==========================================================================
# ==========================================================================
# Multiscale Dynamic Mean Field (MDMF) with local inhibitory plasticity for the Feedback Inhibition Control
#
#  Implemented from:
#  [NaskarEtAl_2018] Amit Naskar, Anirudh Vattikonda, Gustavo Deco,
#      Dipanjan Roy, Arpan Banerjee; Multiscale dynamic mean field (MDMF)
#      model relates resting-state brain dynamics with local cortical
#      excitatory–inhibitory neurotransmitter homeostasis.
#      Network Neuroscience 2021; 5 (3): 757–782.
#      DOI: https://doi.org/10.1162/netn_a_00197
#
# Based on the works by
# [VogelsEtAl_] T. P. Vogels et al., Inhibitory Plasticity Balances Excitation and Inhibition in
#      Sensory Pathways and Memory Networks.Science334,1569-1573(2011).
#      DOI: 10.1126/science.1211095
# [HellyerEtAl_] Peter J. Hellyer, Barbara Jachs, Claudia Clopath, Robert Leech, Local inhibitory
#      plasticity tunes macroscopic brain dynamics and allows the emergence of functional brain
#      networks, NeuroImage,  Volume 124, Part A, 1 January 2016, Pages 85-95
#      DOI: 10.1016/j.neuroimage.2015.08.069
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#      How local excitation-inhibition ratio impacts the whole brain dynamics
#      J. Neurosci., 34 (2014), pp. 7886-7898
#
# By Facundo Faragó and Gustavo Doctorovich
# November 2023
# ==========================================================================
from enum import auto, IntEnum

import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.addr import address_as_void_pointer
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model
from neuronumba.simulator.models import LinearCouplingModel


class Naskar2021(LinearCouplingModel):
    state_vars = Model._build_var_dict(['S_e', 'S_i', 'J'])
    n_state_vars = len(state_vars)
    c_vars = [0]

    observable_vars = Model._build_var_dict(['Ie', 're'])
    n_observable_vars = len(observable_vars)

    t_glu = Attr(default=7.46, attributes=Model.Type.Model)    # concentration of glutamate
    t_gaba = Attr(default=1.82, attributes=Model.Type.Model)   # concentration of GABA
    We = Attr(default=1.0, attributes=Model.Type.Model)        # scaling of external input current to excitatory population
    Wi = Attr(default=0.7, attributes=Model.Type.Model)        # scaling of external input current to inhibitory population
    I0 = Attr(default=0.382, attributes=Model.Type.Model)      #.397  # [nA] overall effective external input
    w = Attr(default=1.4, attributes=Model.Type.Model)         # weight for recurrent self-excitation in each excitatory population
    J_NMDA = Attr(default=0.15, attributes=Model.Type.Model)   # [nA] NMDA current
    M_e = Attr(default=1.0, attributes=Model.Type.Model)
    ae = Attr(default=310.0, attributes=Model.Type.Model)      # [nC^{-1}], g_E in the paper
    be = Attr(default=125.0, attributes=Model.Type.Model)      # = g_E * I^{(E)_{thr}} in the paper = 310 * .403 [nA] = 124.93
    de = Attr(default=0.16, attributes=Model.Type.Model)
    ai = Attr(default=615.0, attributes=Model.Type.Model)      # [nC^{-1}], g_I in the paper
    bi = Attr(default=177.0, attributes=Model.Type.Model)      # = g_I * I^{(I)_{thr}} in the paper = 615 * .288 [nA] = 177.12
    di = Attr(default=0.087, attributes=Model.Type.Model)
    M_i = Attr(default=1.0, attributes=Model.Type.Model)
    alfa_e = Attr(default=0.072, attributes=Model.Type.Model)  # forward rate constant for NMDA gating
    alfa_i = Attr(default=0.53, attributes=Model.Type.Model)   # forward rate constant for GABA gating
    B_e = Attr(default=0.0066, attributes=Model.Type.Model)    # ms^-1  backward rate constant for NMDA gating
    B_i = Attr(default=0.18, attributes=Model.Type.Model)      # ms^-1  backward rate constant for GABA gating
    gamma = Attr(default=1.0, attributes=Model.Type.Model)     # Learning rate
    rho = Attr(default=3.0, attributes=Model.Type.Model)       # target-firing rate of the excitatory population is maintained at the 3 Hz
    I_external = Attr(default=0.0, attributes=Model.Type.Model)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def get_state_vars(self):
        return Naskar2021.state_vars

    @property
    def get_observablevars(self):
        return Naskar2021.observable_vars

    @property
    def get_c_vars(self):
        return Naskar2021.c_vars

    def initial_state(self, n_rois):
        state = np.empty((Naskar2021.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        state[2] = 1.0
        return state

    def initial_observed(self, n_rois):
        observed = np.empty((Naskar2021.n_observable_vars, n_rois))
        observed[0] = 0.0
        observed[1] = 0.0
        return observed

    def get_numba_dfun(self):
        m = self.m.copy()
        # IMPORTANT: you have to add this line here in ALL get_numba_dfun() implementations
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=True)
        def Naskar2021_dfun(state, coupling):                
            Se = np.clip(state[0, :], 0.0, 1.0)
            Si = np.clip(state[1, :], 0.0, 1.0)
            J = state[2, :]

            # Eq for I^E (5). I_external = 0 => resting state condition.
            Ie = m[np.intp(P.We)] * m[np.intp(P.I0)] + m[np.intp(P.w)] * m[np.intp(P.J_NMDA)] * Se + m[np.intp(P.J_NMDA)] * coupling[0, :] - J * Si + m[np.intp(P.I_external)]
            # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
            Ii = m[np.intp(P.Wi)] * m[np.intp(P.I0)] + m[np.intp(P.J_NMDA)] * Se - Si
            y = m[np.intp(P.M_e)] * (m[np.intp(P.ae)] * Ie - m[np.intp(P.be)])
            re = y / (1.0 - np.exp(-m[np.intp(P.de)] * y))
            y = m[np.intp(P.M_i)] * (m[np.intp(P.ai)] * Ii - m[np.intp(P.bi)])
            ri = y / (1.0 - np.exp(-m[np.intp(P.di)] * y))
            # divide by 1000 because we need milliseconds!
            dSe = -Se * m[np.intp(P.B_e)] + m[np.intp(P.alfa_e)] * m[np.intp(P.t_glu)] * (1. - Se) * re / 1000.
            dSi = -Si * m[np.intp(P.B_i)] + m[np.intp(P.alfa_i)] * m[np.intp(P.t_gaba)] * (1. - Si) * ri / 1000.
            dJ = m[np.intp(P.gamma)] * (ri / 1000.0) * (re - m[np.intp(P.rho)]) / 1000.0  # local inhibitory plasticity
            return np.stack((dSe, dSi, dJ)), np.stack((Ie, re))

        return Naskar2021_dfun
