from enum import IntEnum

import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr, AttrType
from neuronumba.numba_tools import address_as_void_pointer
from neuronumba.numba_tools.types import ArrF8_1d, ArrF8_2d
from neuronumba.simulator.models.model import Model


class Hopf(Model):
    state_vars = Model._build_var_dict(['x', 'y'])
    n_state_vars = len(state_vars)
    c_vars = [0, 1]

    observable_vars = Model._build_var_dict([])
    n_observable_vars = len(observable_vars)

    # Model variables
    a = Attr(default=-0.5, attr_type=AttrType.Model)
    omega = Attr(default=0.3, attr_type=AttrType.Model)
    I_external = Attr(default=0.0, attr_type=AttrType.Model)
    conservative = Attr(default=True, attr_type=AttrType.Model)
    weights = Attr(required=True)

    weights_t = Attr(dependant=True)
    sct = Attr(dependant=True)
    ink = Attr(dependant=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights_t = self.weights.T

    @property
    def get_state_vars(self):
        return Hopf.state_vars

    @property
    def get_observablevars(self):
        return Hopf.observable_vars

    @property
    def get_c_vars(self):
        return Hopf.c_vars

    def _init_dependant(self):
        super()._init_dependant()
        self.weights_t = self.weights.T
        self.ink = self.weights_t.sum(axis=1)

    def initial_state(self, n_rois):
        state = np.empty((Hopf.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        return state

    def initial_observed(self, n_rois):
        observed = np.empty((1, 1))
        return observed

    # Hopf model has a non-standard coupling
    def get_numba_coupling(self, g=1.0):
        """
        This is the default coupling for most models, linear coupling using the weights matrix

        :param g: global linear coupling
        :return:
        """
        wt_addr = self.weights_t.ctypes.data
        wt_shape = self.weights_t.shape
        wt_dtype = self.weights_t.dtype
        m_addr = self.m.ctypes.data
        m_shape = self.m.shape
        m_dtype = self.m.dtype
        ink = self.ink

        @nb.njit #(nb.f8[:](nb.f8[:, :], nb.f8[:]))
        def hopf_coupling(weights, state):
            wt = nb.carray(address_as_void_pointer(wt_addr), wt_shape, dtype=wt_dtype)
            m = nb.carray(address_as_void_pointer(m_addr), m_shape, dtype=m_dtype)
            r = wt @ state
            return r - ink * state

        return hopf_coupling

    def get_numba_dfun(self):
        m_addr = self.m.ctypes.data
        m_shape = self.m.shape
        m_dtype = self.m.dtype
        m = self.m
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Hopf_dfun(state: ArrF8_2d, coupling: ArrF8_2d):
            # Comment this line if you deactive @njit for debugging
            m = nb.carray(address_as_void_pointer(m_addr), m_shape, dtype=m_dtype)

            x = state[0, :]
            y = state[1, :]

            pC = m[np.intp(P.I_external)] + 0j
            xcoup = coupling[0, :]  # np.dot(SCT,x) - ink * x  # this is sum(Cij*xi) - sum(Cij)*xj
            ycoup = coupling[1, :]  # np.dot(SCT,y) - ink * y  #
            # Integration step
            dx = (m[np.intp(P.a)] - x ** 2 - y ** 2) * x - m[np.intp(P.omega)] * y + xcoup + pC.real
            dy = (m[np.intp(P.a)] - x ** 2 - y ** 2) * y + m[np.intp(P.omega)] * x + ycoup + pC.imag
            return np.stack((dx, dy)), np.empty((1,1))

        return Hopf_dfun
