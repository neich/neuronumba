import numpy as np
import numba as nb

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import addr
from neuronumba.numba_tools.types import NDA_f8_2d


class History(HasAttr):
    weights = Attr(default=None, required=True)
    # Array with indices of state variables to couple
    c_vars = Attr(default=None, required=True)
    n_cvars = Attr(default=None, required=False, dependant=True)
    n_rois = Attr(default=None, required=False, dependant=True)

    def _init_dependant(self):
        self.c_vars = np.array(self.c_vars, dtype=np.int32)
        self.n_cvars = self.c_vars.shape[0]
        self.n_rois = self.weights.shape[0]


class HistoryDense(History):

    # Global linear coupling
    g = Attr(default=None, required=True)
    delays = Attr(default=None, required=True)
    dt = Attr(required=True)

    i_delays = Attr(dependant=True)
    buffer = Attr(dependant=True)
    n_time = Attr(dependant=True)

    def _init_dependant(self):
        self.i_delays = np.rint(self.delays / self.dt).astype(np.int32)
        self.n_time = np.max(self.i_delays) + 1
        self.buffer = np.zeros((len(self.c_vars), self.n_time, self.n_rois))

    def get_numba_update(self):
        buffer = self.buffer
        n_cvars = self.n_cvars
        c_vars = self.c_vars
        addr = buffer.ctypes.data

        @nb.njit(nb.void(nb.intc, nb.f8[:, :]))
        def c_update(step: nb.intc, state: NDA_f8_2d):
            data = nb.carray(addr.address_as_void_pointer(addr), buffer.shape,
                             dtype=buffer.dtype)
            for i in range(n_cvars):
                data[i, step % self.n_time, :] = state[c_vars[i], :]

        return c_update

    def get_numba_sample(self):
        buffer = self.buffer
        n_time = self.n_time
        weights = self.weights
        i_delays = self.i_delays
        c_vars = self.c_vars
        n_cvars = self.n_cvars
        n_rois = self.n_rois
        g = self.g

        @nb.njit(nb.f8[:, :](nb.intc))
        def h_sample(step):
            time_idx = (step - 1 - i_delays + n_time) % n_time
            result = np.empty((n_cvars, n_rois))
            for v in c_vars:
                delayed_state = np.empty((n_rois, n_rois))
                for i in range(n_rois):
                    delayed_state[i] = buffer[v, time_idx[i], i]
                result[v] = np.sum(weights * delayed_state, axis=0)
            return g * result

        return h_sample




class HistoryNoDelays(History):

    buffer = Attr(dependant=True)


    def _init_dependant(self):
        super()._init_dependant()
        self.buffer = np.zeros((self.n_cvars, self.n_rois), dtype=np.float64)

    def get_numba_sample(self):
        b_addr, b_shape, b_dtype = addr.get_addr(self.buffer)
        # Uncomment this line to debug c_update()
        # b = self.buffer

        # TODO: why adding the signature raises a numba warning about state_coupled being a non contiguous array?
        @nb.njit #(nb.f8[:, :](nb.intc))
        def c_sample(step: nb.intc):
            b = nb.carray(addr.address_as_void_pointer(b_addr), b_shape, dtype=b_dtype)
            return b

        return c_sample

    def get_numba_update(self):
        n_cvars = self.n_cvars
        c_vars = self.c_vars
        b_addr, b_shape, b_dtype = addr.get_addr(self.buffer)
        # Uncomment this line to debug c_update()
        # b = self.buffer

        @nb.njit(nb.void(nb.intc, nb.f8[:, :]))
        def c_update(step: nb.intc, state: NDA_f8_2d):
            b = nb.carray(addr.address_as_void_pointer(b_addr), b_shape, dtype=b_dtype)
            for i in range(n_cvars):
                b[i, :] = state[c_vars[i], :]

        return c_update
