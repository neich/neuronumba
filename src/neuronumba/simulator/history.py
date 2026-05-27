import numpy as np
import numba as nb

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import addr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


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


class HistoryDelays(History):
    """Ring-buffer history with per-(target, source) conduction delays.

    Contract — for compatibility with the simulator's m_couple step:

      `get_numba_sample(step)` returns a (n_cvars, n_rois) array where
      entry [k, i] = g * sum_j W[i, j] * buffer[k, t_idx[i, j], j],
      with t_idx[i, j] = (step - 1 - i_delays[i, j]) mod n_time.

    The per-pair time index is the only thing that distinguishes this from
    the no-delay path; the (n_cvars, n_rois) output shape lets the existing
    DSL `linear` coupling kernel run as an identity passthrough on top.
    """

    g = Attr(default=1.0, required=False)
    # Delay matrix in seconds: delays[i, j] is the time it takes for a
    # signal from region j to arrive at region i.
    delays = Attr(default=None, required=True)
    dt = Attr(required=True)

    i_delays = Attr(dependant=True)
    buffer = Attr(dependant=True)
    n_time = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.i_delays = np.rint(self.delays / self.dt).astype(np.int32)
        # +1 because step indices count from 1; we always need at least one
        # past slot even when all delays round to zero.
        self.n_time = int(np.max(self.i_delays)) + 1
        # Buffer shape: (n_cvars, n_time, n_rois). Initialized to zero —
        # early steps before the buffer fills get implicit zero coupling
        # from far-delay regions, which matches the convention.
        self.buffer = np.zeros(
            (self.n_cvars, self.n_time, self.n_rois), dtype=np.float64
        )

    def get_numba_update(self):
        n_cvars = self.n_cvars
        n_time = self.n_time
        c_vars = self.c_vars
        b_addr, b_shape, b_dtype = addr.get_addr(self.buffer)

        @nb.njit(nb.void(nb.i8, nb.f8[:, :]), cache=NUMBA_CACHE)
        def c_update(step, state):
            buf = addr.create_carray(b_addr, b_shape, b_dtype)
            slot = step % n_time
            for k in range(n_cvars):
                buf[k, slot, :] = state[c_vars[k], :]

        return c_update

    def get_numba_sample(self):
        n_cvars = self.n_cvars
        n_rois = self.n_rois
        n_time = self.n_time
        weights = self.weights
        i_delays = self.i_delays
        g = self.g
        b_addr, b_shape, b_dtype = addr.get_addr(self.buffer)

        @nb.njit(nb.f8[:, :](nb.i8), cache=NUMBA_CACHE)
        def h_sample(step):
            buf = addr.create_carray(b_addr, b_shape, b_dtype)
            result = np.empty((n_cvars, n_rois), dtype=np.float64)
            for k in range(n_cvars):
                for i in range(n_rois):
                    s = 0.0
                    for j in range(n_rois):
                        t_idx = (step - 1 - i_delays[i, j] + n_time) % n_time
                        s += weights[i, j] * buf[k, t_idx, j]
                    result[k, i] = g * s
            return result

        return h_sample


class HistoryNoDelays(History):

    buffer = Attr(dependant=True)


    def _init_dependant(self):
        super()._init_dependant()
        self.buffer = np.zeros((self.n_cvars, self.n_rois), dtype=np.float64)

    def get_numba_sample(self):
        b_addr, b_shape, b_dtype = addr.get_addr(self.buffer)

        # TODO: why adding the signature raises a numba warning about state_coupled being a non contiguous array?
        @nb.njit((nb.f8[:, :])(nb.i8), cache=NUMBA_CACHE)
        def c_sample(step):
            # b = nb.carray(addr.address_as_void_pointer(b_addr), b_shape, dtype=b_dtype)
            b = addr.create_carray(b_addr, b_shape, b_dtype)
            return b

        return c_sample

    def get_numba_update(self):
        n_cvars = self.n_cvars
        c_vars = self.c_vars
        b_addr, b_shape, b_dtype = addr.get_addr(self.buffer)

        @nb.njit(nb.void(nb.i8, nb.f8[:, :]), cache=NUMBA_CACHE)
        def c_update(step, state):
            b = addr.create_carray(b_addr, b_shape, b_dtype)
            for i in range(n_cvars):
                b[i, :] = state[c_vars[i], :]

        return c_update
