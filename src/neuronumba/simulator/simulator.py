import time
import numba as nb
import numpy as np

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.numba_tools import address_as_void_pointer, addr
from neuronumba.numba_tools.types import NDA_f8_2d, NDA_f8_1d
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.simulator.monitors import TemporalAverage
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class Simulator(HasAttr):
    """
    Main simulator class
    """

    connectivity = Attr(required=True)

    model = Attr(required=True)

    integrator = Attr(required=True)

    history = Attr(required=True)

    monitors = Attr(required=True)

    def run(self, t_start=0, t_end=10000, stimulus=None):
        assert self.connectivity, "No connectivity defined for simulation!"

        # Connect all components of the simulation
        self.integrator.configure()
        self.connectivity.configure()
        self.model.configure(weights=self.connectivity.weights)
        self.history.configure(c_vars=self.model.c_vars, weights=self.connectivity.weights)

        dt = self.integrator.dt
        t_max = t_end - t_start

        n_steps = int(t_max / dt)
        n_rois = self.connectivity.n_rois
        init_state = self.model.initial_state(n_rois)
        self._state_shape = (int(self.model.n_state_vars), n_rois)

        for m in self.monitors:
            m.configure(dt=dt, t_max=t_max, n_rois=n_rois)

        m_couple = self.model.get_numba_coupling()
        h_update = self.history.get_numba_update()
        h_sample = self.history.get_numba_sample()
        i_scheme = self.integrator.get_numba_scheme(self.model.get_numba_dfun())
        # TODO: allow more than 1 monitor? Is really useful?
        m_sample = self.monitors[0].get_numba_sample()

        h_update(0, init_state)

        @nb.njit(nb.void(nb.i8, nb.f8[:, :]), cache=NUMBA_CACHE)
        def _sim_loop(n_steps, state):
            for step in range(1, n_steps + 1):
                previous_state_coupled = h_sample(step)
                cpl = m_couple(previous_state_coupled)
                new_state, new_observed = i_scheme(state, cpl)
                h_update(step, new_state)
                m_sample(step-1, new_state, new_observed)
                state = new_state

        start_time = time.perf_counter()
        _sim_loop(n_steps, init_state)
        end_time = time.perf_counter()
        return end_time - start_time


# =====================================================================================
# Convenience method to put all components together
# =====================================================================================
def simulate_nodelay(model, integrator, weights, obs_var, sampling_period, t_max_neuronal, t_warmup):
    n_rois = weights.shape[0]
    lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
    speed = 1.0
    con = Connectivity(weights=weights, lengths=lengths, speed=speed)
    history = HistoryNoDelays()
    monitor = TemporalAverage(period=sampling_period, monitor_vars=model.get_var_info([obs_var]))
    s = Simulator(connectivity=con, model=model, history=history, integrator=integrator, monitors=[monitor])
    t = s.run(0, t_warmup + t_max_neuronal)
    data = monitor.data(obs_var)
    data_from = int(data.shape[0] * t_warmup / (t_max_neuronal + t_warmup))
    return data[data_from:, :]
