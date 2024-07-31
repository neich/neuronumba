import numpy as np
import numba as nb

from src.neuronumba import HasAttr, Attr
from src.neuronumba import address_as_void_pointer
from src.neuronumba import ArrF8_2d


class Monitor(HasAttr):
    dt = Attr(required=True)
    n_rois = Attr(required=True)
    
    state_vars = Attr(required=True)
    obs_vars = Attr(required=True)
    state_vars_indices = Attr(dependant=True)
    obs_vars_indices = Attr(dependant=True)

    n_state_vars = Attr(dependant=True)
    n_obs_vars = Attr(dependant=True)

    def _init_dependant(self):
        # Extract index list from input dictionary
        s_vars_i = [i for _, (i, _) in self.state_vars.items()]
        o_var_i = [i for _, (i, _) in self.obs_vars.items()]

        self.n_state_vars = len(s_vars_i)
        self.n_obs_vars = len(o_var_i)
        self.state_vars_indices = np.array(s_vars_i, dtype=np.int32)
        self.obs_vars_indices = np.array(o_var_i, dtype=np.int32)

    def data(self, var: str):
        if var in self.state_vars:
            return self._get_data_state(self.state_vars[var][1])
        elif var in self.obs_vars:
            return self._get_data_obs(self.obs_vars[var][1])
        else:
            raise Exception(f"Variable <{var}> not defined in monitor!")

    # Methods to be implemented in subclasses

    def sample(self, step, state, observed):
        raise NotImplementedError

    def _get_data_state(self, index: int):
        raise NotImplementedError

    def _get_data_obs(self, index: int):
        raise NotImplementedError


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

    def _get_data_state(self, index: int):
        return self.buffer_state[:, index, :]

    def _get_data_obs(self, index: int):
        return self.buffer_observed[:, index, :]

    def get_numba_sample(self):
        buffer_state = self.buffer_state
        addr_state = buffer_state.ctypes.data
        state_vars = self.state_vars_indices
        n_state = nb.intc(self.n_state_vars)
        buffer_observed = self.buffer_observed
        addr_observed = buffer_observed.ctypes.data
        obs_vars = self.obs_vars_indices
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