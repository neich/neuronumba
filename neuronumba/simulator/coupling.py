from numba import f8, int32, njit, intc, void
import numpy as np
import numba as nb

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import address_as_void_pointer
from neuronumba.numba_tools.types import ArrF8_2d


class Coupling(HasAttr):
    weights = Attr(default=None, required=True)
    # Array with indices of state variables to couple
    c_vars = Attr(default=None, required=True)
    n_cvars = Attr(default=None, required=False, dependant=True)
    n_rois = Attr(default=None, required=False, dependant=True)

    def _init_dependant(self):
        self.c_vars = np.array(self.c_vars, dtype=np.int32)
        self.n_cvars = self.c_vars.shape[0]
        self.n_rois = self.weights.shape[0]


class CouplingLinearDense(Coupling):

    # Global linear coupling
    g = Attr(default=None, required=True)
    delays = Attr(default=None, required=True)

    i_delays = Attr(dependant=True)
    buffer = Attr(dependant=True)
    n_time = Attr(dependant=True)

    def _init_dependantt(self):
        self.i_delays = np.rint(self.delays / dt).astype(np.int32)
        self.n_time = np.max(self.i_delays) + 1
        self.buffer = np.zeros((len(self.c_vars), self.n_time, self.n_rois))

    def update(self, step, state):
        self.buffer[:, step % self.n_time, :] = state[self.c_vars]

    def couple(self, step):
        return CouplingLinearDense_couple(step, self.buffer, self.n_time, self.weights, self.i_delays, 
                                          self.c_vars, self.n_cvars, self.n_rois)

    def get_numba_couple(self):
        buffer = self.buffer
        n_time = self.n_time
        weights = self.weights
        i_delays = self.i_delays
        c_vars = self.c_vars
        n_cvars = self.n_cvars
        n_rois = self.n_rois
        @njit
        def c_couple(step):
            return CouplingLinearDense_couple(step, buffer, n_time, weights, i_delays, c_vars, n_cvars, n_rois)

        return c_couple


@njit(f8[:,:](intc, f8[:,:,:], intc, f8[:,:], int32[:,:], int32[:], intc, intc))
def CouplingLinearDense_couple(step, buffer, n_time, weights, i_delays, c_vars, n_cvars, n_rois):
    time_idx = (step - 1 - i_delays + n_time) % n_time
    result = np.empty((n_cvars, n_rois))
    for v in c_vars:
        delayed_state = np.empty((n_rois, n_rois))
        for i in range(n_rois):
            delayed_state[i] = buffer[v, time_idx[i], i]
        result[v] = np.sum(weights * delayed_state, axis=0)
    return result


class CouplingLinearNoDelays(Coupling):

    # Global linear coupling
    g = Attr(default=None, required=True)

    n_cvars = Attr(dependant=True)
    n_rois = Attr(dependant=True)
    buffer = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.buffer = np.zeros((1, self.n_cvars, self.n_rois), dtype=np.float64)

    def get_numba_couple(self):
        buffer = self.buffer
        addr = buffer.ctypes.data
        weights = self.weights
        n_cvars = self.n_cvars
        n_rois = self.n_rois

        @njit(f8[:, :](intc))
        def c_couple(step: intc):
            data = nb.carray(address_as_void_pointer(addr), buffer.shape,
                             dtype=buffer.dtype)
            result = np.empty((n_cvars, n_rois), dtype=np.float64)
            for i in range(n_cvars):
                r = weights @ data[0, i, :]
                result[i, :] = r
            return result

        return c_couple

    def get_numba_update(self):
        buffer = self.buffer
        n_cvars = self.n_cvars
        c_vars = self.c_vars
        addr = buffer.ctypes.data

        @njit(void(intc, f8[:, :]))
        def c_update(step: intc, state: ArrF8_2d):
            data = nb.carray(address_as_void_pointer(addr), buffer.shape,
                             dtype=buffer.dtype)
            for i in range(n_cvars):
                data[0, i, :] = state[c_vars[i], :]

        return c_update
