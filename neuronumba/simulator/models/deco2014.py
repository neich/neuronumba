# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
#
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
# ==========================================================================
import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools import address_as_void_pointer
from neuronumba.numba_tools.types import ArrF8_1d, ArrF8_2d
from neuronumba.simulator.models import Model


# ====================== var indices into the array
t_glu = 0
t_gaba = 1
We = 2
Wi = 3
I0 = 4
w = 5
J_NMDA = 6
M_e = 7
ae = 8
be = 9
de = 10
M_i = 11
ai = 12
bi = 13
di = 14
alfa_i = 15
alfa_e = 16
B_e = 17
B_i = 18
J = 19
I_external = 20


# ==========================================================================
class Deco2014(Model):
    n_params = 21
    state_vars = Model._build_var_dict(['S_e', 'S_i'])
    n_state_vars = len(state_vars)
    c_vars = [0]

    observable_vars = Model._build_var_dict(['Ie', 're'])
    n_observable_vars = len(observable_vars)

    t_glu = Attr(default=7.46)
    t_gaba = Attr(default=1.82)
    We = Attr(default=1.0)
    Wi = Attr(default=0.7)
    I0 = Attr(default=0.382)
    w = Attr(default=1.4)
    J_NMDA = Attr(default=0.15)
    M_e = Attr(default=1.0)
    ae = Attr(default=310.0)
    be = Attr(default=125.0)
    de = Attr(default=0.16)
    ai = Attr(default=615.0)
    bi = Attr(default=177.0)
    di = Attr(default=0.087)
    M_i = Attr(default=1.0)
    alfa_e = Attr(default=0.072)
    alfa_i = Attr(default=0.53)
    B_e = Attr(default=0.0066)
    B_i = Attr(default=0.18)
    J = Attr(default=1.0)
    I_external = Attr(default=0.0)

    m = Attr(dependant=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def get_state_vars(self):
        return Deco2014.state_vars

    @property
    def get_observablevars(self):
        return Deco2014.observable_vars

    @property
    def get_c_vars(self):
        return Deco2014.c_vars

    def _init_dependant(self):
        self.m = np.empty((Deco2014.n_params, self.n_rois))
        self.m[t_glu] = self.as_array(self.t_glu)
        self.m[t_gaba] = self.as_array(self.t_gaba)
        self.m[We] = self.as_array(self.We)
        self.m[Wi] = self.as_array(self.Wi)
        self.m[I0] = self.as_array(self.I0)
        self.m[w] = self.as_array(self.w)
        self.m[J_NMDA] = self.as_array(self.J_NMDA)
        self.m[M_e] = self.as_array(self.M_e)
        self.m[ae] = self.as_array(self.ae)
        self.m[be] = self.as_array(self.be)
        self.m[de] = self.as_array(self.de)
        self.m[ai] = self.as_array(self.ai)
        self.m[bi] = self.as_array(self.bi)
        self.m[di] = self.as_array(self.di)
        self.m[M_i] = self.as_array(self.M_i)
        self.m[alfa_e] = self.as_array(self.alfa_e)
        self.m[alfa_i] = self.as_array(self.alfa_i)
        self.m[B_e] = self.as_array(self.B_e)
        self.m[B_i] = self.as_array(self.B_i)
        self.m[J] = self.as_array(self.J)
        self.m[I_external] = self.as_array(self.I_external)

    def initial_state(self, n_rois):
        state = np.empty((Deco2014.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        return state

    def initial_observed(self, n_rois):
        observed = np.empty((Deco2014.n_observable_vars, n_rois))
        observed[0] = 0.0
        observed[1] = 0.0
        return observed

    def get_numba_dfun(self):
        m_addr = self.m.ctypes.data
        m_shape = self.m.shape
        m_dtype = self.m.dtype
        # Uncomment this line if you want to debug Deco2014_dfun without @nb.njit
        # m = self.m

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Deco2014_dfun(state: ArrF8_2d, coupling: ArrF8_2d):
            # Comment this line if you deactivate @nb.njit for debugging
            m = nb.carray(address_as_void_pointer(m_addr), m_shape, dtype=m_dtype)

            Se = np.clip(state[0, :],0.0, 1.0)
            Si = np.clip(state[1, :], 0.0,1.0)

            # Eq for I^E (5). I_external = 0 => resting state condition.
            Ie = m[We] * m[I0] + m[w] * m[J_NMDA] * Se + m[J_NMDA] * coupling[0, :] - m[J] * Si + m[I_external]
            # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
            Ii = m[Wi] * m[I0] + m[J_NMDA] * Se - Si
            y = m[M_e] * (m[ae] * Ie - m[be])
            re = y / (1.0 - np.exp(-m[de] * y))
            y = m[M_i] * (m[ai] * Ii - m[bi])
            ri = y / (1.0 - np.exp(-m[di] * y))
            # divide by 1000 because we need milliseconds!
            dSe = -Se * m[B_e] + m[alfa_e] * m[t_glu] * (1. - Se) * re / 1000.
            dSi = -Si * m[B_i] + m[alfa_i] * m[t_gaba] * (1. - Si) * ri / 1000.
            return np.stack((dSe, dSi)), np.stack((Ie, re))

        return Deco2014_dfun
