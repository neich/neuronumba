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

taon = 0
taog = 1
gamma_e = 2
gamma_i = 3
J_NMDA = 4
I0 = 5
Jext_e = 6
Jext_i = 7
ae = 8
be = 9
de = 10
ai = 12
bi = 13
di = 14
w = 15
J = 16
I_external = 17


class Deco2014(Model):
    n_params = 18
    state_vars = Model._build_var_dict(['S_e', 'S_i'])
    n_state_vars = len(state_vars)
    c_vars = [0]

    observable_vars = Model._build_var_dict(['Ie', 're'])
    n_observable_vars = len(observable_vars)

    taon = Attr(default=100.0)
    taog = Attr(default=10.0)
    gamma_e = Attr(default=0.641)
    gamma_i = Attr(default=1.0)
    I0 = Attr(default=0.382)
    w = Attr(default=1.4)
    J_NMDA = Attr(default=0.15)
    Jext_e = Attr(default=1.0)
    Jext_i = Attr(default=0.7)
    ae = Attr(default=310.0)
    be = Attr(default=125.0)
    de = Attr(default=0.16)
    ai = Attr(default=615.0)
    bi = Attr(default=177.0)
    di = Attr(default=0.087)
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
        self.m[taon] = self.as_array(self.taon)
        self.m[taog] = self.as_array(self.taog)
        self.m[gamma_e] = self.as_array(self.gamma_e)
        self.m[gamma_i] = self.as_array(self.gamma_i)
        self.m[I0] = self.as_array(self.I0)
        self.m[w] = self.as_array(self.w)
        self.m[J_NMDA] = self.as_array(self.J_NMDA)
        self.m[Jext_e] = self.as_array(self.Jext_e)
        self.m[ae] = self.as_array(self.ae)
        self.m[be] = self.as_array(self.be)
        self.m[de] = self.as_array(self.de)
        self.m[ai] = self.as_array(self.ai)
        self.m[bi] = self.as_array(self.bi)
        self.m[di] = self.as_array(self.di)
        self.m[Jext_i] = self.as_array(self.Jext_i)
        self.m[w] = self.as_array(self.w)
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

            Se = np.clip(state[0, :],0.0,1.0)
            Si = np.clip(state[1, :],0.0,1.0)

            # Eq for I^E (5). I_external = 0 => resting state condition.
            Ie = m[Jext_e] * m[I0] + m[w] * m[J_NMDA] * Se + m[J_NMDA] * coupling[0, :] - m[J] * Si + m[I_external]
            # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
            Ii = m[Jext_i] * m[I0] + m[J_NMDA] * Se - Si
            y = m[ae] * Ie - m[be]
            re = y / (1.0 - np.exp(-m[de] * y))
            y = m[ai] * Ii - m[bi]
            ri = y / (1.0 - np.exp(-m[di] * y))
            # divide by 1000 because we need milliseconds!
            dSe = -Se / m[taon] + m[gamma_e] * (1. - Se) * re / 1000.
            dSi = -Si / m[taog] + m[gamma_i] * ri / 1000.
            return np.stack((dSe, dSi)), np.stack((Ie, re))

        return Deco2014_dfun
