import numpy as np
import numba as nb

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import address_as_void_pointer
from neuronumba.numba_tools.types import ArrF8_2d


class Monitor(HasAttr):
    dt = Attr(required=True)
    n_rois = Attr(required=True)
    
    state_vars = Attr(required=True)
    obs_vars = Attr(required=True)
    
    n_state_vars = Attr(dependant=True)
    n_obs_vars = Attr(dependant=True)

    def _init_dependant(self):
        self.n_state_vars = len(self.state_vars)
        self.n_obs_vars = len(self.obs_vars)
        self.state_vars = np.array(self.state_vars, dtype=np.int32)
        self.obs_vars = np.array(self.obs_vars, dtype=np.int32)

    def sample(self, step, state, observed):
        pass

    def data_state(self):
        pass

    def data_observed(self):
        pass


class RawMonitor(Monitor):

    buffer = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.buffer = []

    def sample(self, step, state, observed):
        self.buffer.append(state)

    def data(self):
        return np.array(self.buffer)


class RawSubSample(Monitor):
    period = Attr(default=None, required=True)
    t_max = Attr(default=None, required=True)

    n_interim_samples = Attr(dependant=True)
    buffer_state = Attr(dependant=True)
    buffer_observed = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.n_interim_samples = int(self.period / self.dt)
        time_samples = 1 + int(self.t_max / self.period)
        if self.n_state_vars:
            self.buffer_state = np.empty((time_samples, self.n_state_vars, self.n_rois))
        else:
            self.buffer_state = np.empty(1, )
        if self.n_obs_vars:
            self.buffer_observed = np.empty((time_samples, self.n_obs_vars, self.n_rois))
        else:
            self.buffer_observed = np.empty(1, )

    def data_state(self):
        return self.buffer_state

    def data_observed(self):
        return self.buffer_observed

    def get_numba_sample(self):
        buffer_state = self.buffer_state
        addr_state = buffer_state.ctypes.data
        state_vars = self.state_vars
        n_state = nb.intc(self.n_state_vars)
        buffer_observed = self.buffer_observed
        addr_observed = buffer_observed.ctypes.data
        obs_vars = self.obs_vars
        n_obs = nb.intc(self.n_obs_vars)
        n_interim_samples = nb.intc(self.n_interim_samples)

        @nb.njit(nb.void(nb.intc, nb.f8[:, :], nb.f8[:, :]))
        def m_sample(step: nb.intc, state: ArrF8_2d, observed: ArrF8_2d):
            if step % n_interim_samples == 0:
                if n_state > 0:
                    bnb_state = nb.carray(address_as_void_pointer(addr_state), buffer_state.shape,
                                          dtype=buffer_state.dtype)
                    i = nb.intc(step / n_interim_samples)
                    for v in range(n_state):
                        bnb_state[i, v, :] = state[state_vars[v], :]
                if n_obs > 0:
                    bnb_obs = nb.carray(address_as_void_pointer(addr_observed), buffer_observed.shape,
                                        dtype=buffer_observed.dtype)
                    i = nb.intc(step / n_interim_samples)
                    for v in range(n_obs):
                        bnb_obs[i, v, :] = observed[obs_vars[v], :]

        return m_sample


class TemporalAverage(Monitor):
    period = Attr(default=None, required=True)
    t_max = Attr(default=None, required=True)

    n_interim_samples = Attr(dependant=True)
    buffer_state = Attr(dependant=True)
    buffer_observed = Attr(dependant=True)

    i_buffer_state = Attr(dependant=True)
    i_buffer_observed = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.n_interim_samples = int(self.period / self.dt)
        time_samples = 1 + int(self.t_max / self.period)
        if self.n_state_vars:
            self.i_buffer_state = np.zeros((self.n_interim_samples, self.n_state_vars, self.n_rois))
            self.buffer_state = np.empty((time_samples, self.n_state_vars, self.n_rois))
        else:
            self.i_buffer_state = np.empty(1, )
            self.buffer_state = np.empty(1, )
        if self.n_obs_vars:
            self.i_buffer_observed = np.zeros((self.n_interim_samples, self.n_state_vars, self.n_rois))
            self.buffer_observed = np.empty((time_samples, self.n_obs_vars, self.n_rois))
        else:
            self.i_buffer_observed = np.empty(1, )
            self.buffer_observed = np.empty(1, )

    def data_state(self):
        return self.buffer_state

    def data_observed(self):
        return self.buffer_observed

    def get_numba_sample(self):
        buffer_state = self.buffer_state
        i_buffer_state = self.i_buffer_state
        addr_state = buffer_state.ctypes.data
        addr_i_state = i_buffer_state.ctypes.data
        state_vars = self.state_vars
        n_state = nb.intc(self.n_state_vars)
        buffer_observed = self.buffer_observed
        i_buffer_observed = self.i_buffer_observed
        addr_observed = buffer_observed.ctypes.data
        addr_i_observed = i_buffer_observed.ctypes.data
        obs_vars = self.obs_vars
        n_obs = nb.intc(self.n_obs_vars)
        n_interim_samples = nb.intc(self.n_interim_samples)

        @nb.njit(nb.void(nb.intc, nb.f8[:, :], nb.f8[:, :]))
        def m_sample(step: nb.intc, state: ArrF8_2d, observed: ArrF8_2d):
            # Update interim buffer
            if n_state > 0:
                i_bnb_state = nb.carray(address_as_void_pointer(addr_i_state), i_buffer_state.shape,
                                        dtype=i_buffer_state.dtype)
                i_bnb_state[(step - 1) % self.n_interim_samples] = state
                if step % n_interim_samples == 0:
                    bnb_state = nb.carray(address_as_void_pointer(addr_state), buffer_state.shape,
                                          dtype=buffer_state.dtype)
                    i = nb.intc(step / n_interim_samples)
                    for v in range(n_state):
                        bnb_state[i, v, :] = np.average(i_bnb_state[:, state_vars[v], :], axis=0)

            if n_obs > 0:
                i_bnb_observed = nb.carray(address_as_void_pointer(addr_i_observed), i_buffer_observed.shape,
                                        dtype=i_buffer_observed.dtype)
                i_bnb_observed[(step - 1) % self.n_interim_samples] = observed
                if step % n_interim_samples == 0:
                    bnb_observed = nb.carray(address_as_void_pointer(addr_observed), buffer_observed.shape,
                                          dtype=buffer_observed.dtype)
                    i = nb.intc(step / n_interim_samples)
                    for v in range(n_obs):
                        bnb_observed[i, v, :] = np.average(i_bnb_observed[:, obs_vars[v], :], axis=0)

        return m_sample

# class TemporalAverage(Monitor):
#     n_interim_samples = Attr(dependant=True)
#     buffer = Attr(dependant=True)
#     interim_buffer = Attr(dependant=True)
#
#     def _init_dependant(self):
#         super()._init_dependant()
#         self.n_interim_samples = int(self.period / self.dt)
#         self.interim_buffer = np.zeros((self.n_interim_samples, shape[0], shape[1]))
#
#     def sample(self, step, state):
#         self.interim_buffer[(step - 1) % self.n_interim_samples] = state
#         if step % self.n_interim_samples == 0:
#             self.buffer.append(np.average(self.interim_buffer, axis=0))
#
#     def data(self):
#         return np.array(self.buffer)
