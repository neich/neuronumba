import numpy as np
import numba as nb
import math

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import addr
from neuronumba.numba_tools.addr import address_as_void_pointer
from neuronumba.numba_tools.types import NDA_f8_2d

class Monitor(HasAttr):
    dt = Attr(required=True)
    n_rois = Attr(required=True)
    
    monitor_vars = Attr(required=True)
    state_vars_indices = Attr(dependant=True)
    obs_vars_indices = Attr(dependant=True)

    n_state_vars = Attr(dependant=True)
    n_obs_vars = Attr(dependant=True)

    def _init_dependant(self):
        # Extract index list from input dictionary
        s_vars_i = []
        o_var_i = []

        for v in self.monitor_vars:
            if self.monitor_vars[v][0]:
                s_vars_i.append(self.monitor_vars[v][2])
            else:
                o_var_i.append(self.monitor_vars[v][2])

        self.n_state_vars = len(s_vars_i)
        self.n_obs_vars = len(o_var_i)
        self.state_vars_indices = np.array(s_vars_i, dtype=np.int32)
        self.obs_vars_indices = np.array(o_var_i, dtype=np.int32)

    def data(self, var: str):
        if var not in self.monitor_vars:
            raise Exception(f"Variable <{var}> not defined in monitor!")
        value = self.monitor_vars[var]
        if value[0]:
            return self._get_data_state(value[1])
        else:
            return self._get_data_obs(value[1])

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
        n_steps = int(self.t_max / self.dt)
        time_samples = 1 + int(n_steps / self.n_interim_samples)
        if self.n_state_vars:
            self.buffer_state = np.zeros((time_samples, self.n_state_vars, self.n_rois))
        else:
            self.buffer_state = np.empty(1, )
        if self.n_obs_vars:
            self.buffer_observed = np.zeros((time_samples, self.n_obs_vars, self.n_rois))
        else:
            self.buffer_observed = np.empty(1, )

    def _get_data_state(self, index: int):
        return self.buffer_state[:, index, :]

    def _get_data_obs(self, index: int):
        return self.buffer_observed[:, index, :]

    def get_numba_sample(self):
        bs = self.buffer_state
        bs_addr, bs_shape, bs_dtype = addr.get_addr(bs)
        state_vars = self.state_vars_indices
        n_state = nb.intc(self.n_state_vars)
        bo = self.buffer_observed
        bo_addr, bo_shape, bo_dtype = addr.get_addr(bo)
        obs_vars = self.obs_vars_indices
        n_obs = nb.intc(self.n_obs_vars)
        n_interim_samples = nb.intc(self.n_interim_samples)

        @nb.njit(nb.void(nb.intc, nb.f8[:, :], nb.f8[:, :]))
        def m_sample(step: nb.intc, state: NDA_f8_2d, observed: NDA_f8_2d):
            if step % n_interim_samples == 0:
                if n_state > 0:
                    bs = nb.carray(address_as_void_pointer(bs_addr), bs_shape, dtype=bs_dtype)
                    i = int(step / n_interim_samples)
                    for v in range(n_state):
                        bs[i, v, :] = state[state_vars[v], :]
                if n_obs > 0:
                    bo = nb.carray(address_as_void_pointer(bo_addr), bo_shape, dtype=bo_dtype)
                    i = int(step / n_interim_samples)
                    for v in range(n_obs):
                        bo[i, v, :] = observed[obs_vars[v], :]

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
        # Okey this give problems, I think the +1 is not ok when resto modulo is zero.
        # Because I'm not sure I can remove it, I will put the math.ceil. If this cause problems
        # in divisions (ex. t_max / perios) that are not exact, remove the ceil and leave just the
        # int(self.t_max / self.perios). 
        # The following was the line causing problems:
        # time_samples = 1 + int(self.t_max / self.period)
        # And this the proposal:
        time_samples = math.ceil(self.t_max / self.period)
        if self.n_state_vars:
            self.i_buffer_state = np.zeros((self.n_interim_samples, self.n_state_vars, self.n_rois))
            self.buffer_state = np.empty((time_samples, self.n_state_vars, self.n_rois))
        else:
            self.i_buffer_state = np.empty(1, )
            self.buffer_state = np.empty(1, )
        if self.n_obs_vars:
            self.i_buffer_observed = np.zeros((self.n_interim_samples, self.n_obs_vars, self.n_rois))
            self.buffer_observed = np.empty((time_samples, self.n_obs_vars, self.n_rois))
        else:
            self.i_buffer_observed = np.empty(1, )
            self.buffer_observed = np.empty(1, )

    def data_state(self):
        return self.buffer_state

    def data_observed(self):
        return self.buffer_observed

    def _get_data_state(self, index: int):
        return self.buffer_state[:, index, :]

    def _get_data_obs(self, index: int):
        return self.buffer_observed[:, index, :]

    def get_numba_sample(self):
        buffer_state = self.buffer_state
        bs_addr, bs_shape, bs_dtype = addr.get_addr(buffer_state)
        i_buffer_state = self.i_buffer_state
        ibs_addr, ibs_shape, ibs_dtype = addr.get_addr(i_buffer_state)
        state_vars = self.state_vars_indices
        n_state = nb.intc(self.n_state_vars)
        buffer_observed = self.buffer_observed
        bo_addr, bo_shape, bo_dtype = addr.get_addr(buffer_observed)
        i_buffer_observed = self.i_buffer_observed
        ibo_addr, ibo_shape, ibo_dtype = addr.get_addr(i_buffer_observed)
        obs_vars = self.obs_vars_indices
        n_obs = nb.intc(self.n_obs_vars)
        n_interim_samples = nb.intc(self.n_interim_samples)

        @nb.njit(nb.void(nb.intc, nb.f8[:, :], nb.f8[:, :]))
        def m_sample(step, state, observed):
            # Update interim buffer
            if n_state > 0:
                # i_bnb_state = nb.carray(address_as_void_pointer(ibs_addr), ibs_shape, dtype=ibs_dtype)
                i_bnb_state = addr.create_carray(ibs_addr, ibs_shape, ibs_dtype)
                # i_bnb_state = self.i_buffer_state
                i_bnb_state[(step - 1) % n_interim_samples] = state[state_vars, :]
                if step % n_interim_samples == 0:
                    # bnb_state = nb.carray(address_as_void_pointer(bs_addr), bs_shape, dtype=bs_dtype)
                    bnb_state = addr.create_carray(bs_addr, bs_shape, bs_dtype)
                    # bnb_state = self.buffer_state
                    i = nb.intc(step / n_interim_samples)
                    bnb_state[i, :, :] = i_bnb_state.sum(axis=0) / ibs_shape[0]

            if n_obs > 0:
                # i_bnb_observed = nb.carray(address_as_void_pointer(ibo_addr), ibo_shape, dtype=ibo_dtype)
                i_bnb_observed = addr.create_carray(ibo_addr, ibo_shape, ibo_dtype)
                # i_bnb_observed = self.i_buffer_observed
                i_bnb_observed[(step - 1) % n_interim_samples] = observed[obs_vars, :]
                if step % n_interim_samples == 0:
                    # bnb_observed = nb.carray(address_as_void_pointer(bo_addr), bo_shape, dtype=bo_dtype)
                    bnb_observed = addr.create_carray(bo_addr, bo_shape, bo_dtype)
                    # bnb_observed = self.buffer_observed
                    i = nb.intc(step / n_interim_samples)
                    bnb_observed[i, :, :] = i_bnb_observed.sum(axis=0) / ibo_shape[0]

        return m_sample