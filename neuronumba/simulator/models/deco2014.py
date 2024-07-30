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

from neuronumba.basic.attr import Attr, AttrType
from neuronumba.numba_tools import address_as_void_pointer
from neuronumba.numba_tools.types import ArrF8_1d, ArrF8_2d
from neuronumba.simulator.models import Model
from neuronumba.simulator.models.model import LinearCouplingModel


class Deco2014(LinearCouplingModel):
    state_vars = Model._build_var_dict(['S_e', 'S_i'])
    n_state_vars = len(state_vars)
    c_vars = [0]

    observable_vars = Model._build_var_dict(['Ie', 're'])
    n_observable_vars = len(observable_vars)

    taon = Attr(default=100.0, attr_type=AttrType.Model)
    taog = Attr(default=10.0, attr_type=AttrType.Model)
    gamma_e = Attr(default=0.641, attr_type=AttrType.Model)
    gamma_i = Attr(default=1.0, attr_type=AttrType.Model)
    I0 = Attr(default=0.382, attr_type=AttrType.Model)
    w = Attr(default=1.4, attr_type=AttrType.Model)
    J_NMDA = Attr(default=0.15, attr_type=AttrType.Model)
    Jext_e = Attr(default=1.0, attr_type=AttrType.Model)
    Jext_i = Attr(default=0.7, attr_type=AttrType.Model)
    ae = Attr(default=310.0, attr_type=AttrType.Model)
    be = Attr(default=125.0, attr_type=AttrType.Model)
    de = Attr(default=0.16, attr_type=AttrType.Model)
    ai = Attr(default=615.0, attr_type=AttrType.Model)
    bi = Attr(default=177.0, attr_type=AttrType.Model)
    di = Attr(default=0.087, attr_type=AttrType.Model)
    J = Attr(default=1.0, attr_type=AttrType.Model)
    I_external = Attr(default=0.0, attr_type=AttrType.Model)

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
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Deco2014_dfun(state: ArrF8_2d, coupling: ArrF8_2d):
            # Comment this line if you deactivate @nb.njit for debugging
            m = nb.carray(address_as_void_pointer(m_addr), m_shape, dtype=m_dtype)

            Se = np.clip(state[0, :],0.0,1.0)
            Si = np.clip(state[1, :],0.0,1.0)

            # Eq for I^E (5). I_external = 0 => resting state condition.
            Ie = m[np.intp(P.Jext_e)] * m[np.intp(P.I0)] + m[np.intp(P.w)] * m[np.intp(P.J_NMDA)] * Se + m[np.intp(P.J_NMDA)] * coupling[0, :] - m[np.intp(P.J)] * Si + m[np.intp(P.I_external)]
            # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
            Ii = m[np.intp(P.Jext_i)] * m[np.intp(P.I0)] + m[np.intp(P.J_NMDA)] * Se - Si
            y = m[np.intp(P.ae)] * Ie - m[np.intp(P.be)]
            re = y / (1.0 - np.exp(-m[np.intp(P.de)] * y))
            y = m[np.intp(P.ai)] * Ii - m[np.intp(P.bi)]
            ri = y / (1.0 - np.exp(-m[np.intp(P.di)] * y))
            # divide by 1000 because we need milliseconds!
            dSe = -Se / m[np.intp(P.taon)] + m[np.intp(P.gamma_e)] * (1. - Se) * re / 1000.
            dSi = -Si / m[np.intp(P.taog)] + m[np.intp(P.gamma_i)] * ri / 1000.
            return np.stack((dSe, dSi)), np.stack((Ie, re))

        return Deco2014_dfun
