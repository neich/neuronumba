import numpy as np
import numba as nb
from numba import float64, intc, void

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import address_as_void_pointer
from neuronumba.numba_tools.types import ArrF8_2d


class Monitor(HasAttr):
    dt = Attr(default=None, required=True)
    shape = Attr(default=None, required=True)

    def sample(self, step, state):
        pass


class RawMonitor(Monitor):

    buffer = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.buffer = []

    def sample(self, step, state):
        self.buffer.append(state)

    def data(self):
        return np.array(self.buffer)


class RawSubSample(Monitor):
    period = Attr(default=None, required=True)
    t_max = Attr(default=None, required=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_interim_samples = 0
        self.buffer = []

    def configure(self, **kwargs):
        super().configure(**kwargs)
        self._check_required()
        self.n_interim_samples = int(self.period / self.dt)
        self.buffer = np.empty((1 + int(self.t_max / self.period), self.shape[0], self.shape[1]))

    def sample(self, step, state):
        if step % self.n_interim_samples == 0:
            self.buffer[self.n_interim_samples] = state

    def data(self):
        return np.array(self.buffer)

    def get_numba_sample(self):
        buffer = self.buffer
        addr = buffer.ctypes.data
        n_interim_samples = intc(self.n_interim_samples)

        @nb.njit(void(intc, float64[:, :]))
        def m_sample(step: intc, state: ArrF8_2d):
            data = nb.carray(address_as_void_pointer(addr), buffer.shape,
                             dtype=buffer.dtype)
            if step % n_interim_samples == 0:
                data[intc(step / n_interim_samples)] = state

        return m_sample


class TemporalAverage(Monitor):
    n_interim_samples = Attr(dependant=True)
    buffer = Attr(dependant=True)
    interim_buffer = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.n_interim_samples = int(self.period / self.dt)
        self.interim_buffer = np.zeros((self.n_interim_samples, shape[0], shape[1]))

    def sample(self, step, state):
        self.interim_buffer[(step - 1) % self.n_interim_samples] = state
        if step % self.n_interim_samples == 0:
            self.buffer.append(np.average(self.interim_buffer, axis=0))

    def data(self):
        return np.array(self.buffer)
