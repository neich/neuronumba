from typing import Any
import numpy as np

from numba import njit, void, intc, f8, i8
from nptyping import NDArray, Shape

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.numba_tools.types import ArrF8_2d, ArrF8_1d


class Simulator(HasAttr):
    connectivity = Attr(
        default=None,
        required=True)

    model = Attr(
        default=None,
        required=True)

    integrator = Attr(
        default=None,
        required=True)

    coupling = Attr(
        default=None,
        required=True)

    monitors = Attr(
        default=None,
        required=True
    )

    def run(self, t_start=0, t_end=10000, stimulus=None):
        assert self.connectivity, "No connectivity defined for simulation!"

        self.integrator.configure()
        self.coupling.configure()
        self.connectivity.configure()
        self.model.configure()

        dt = self.integrator.dt
        t_max = t_end - t_start

        n_steps = int((t_end - t_start) / dt)
        n_rois = self.connectivity.n_rois
        state = self.model.initial_state(n_rois)
        self._state_shape = (int(self.model.n_state_vars), n_rois)

        for m in self.monitors:
            m.configure(shape=state.shape, dt=dt, t_max=t_max)

        c_couple = self.coupling.get_numba_couple()
        c_update = self.coupling.get_numba_update()
        i_scheme = self.integrator.get_numba_scheme(self.model.dfun)
        m_sample = self.monitors[0].get_numba_sample()

        c_update(0, state)

        @njit(void(intc,  # n_steps
                   f8[:, :],  # state
                   f8[:],  # model
                   )
              )
        def _sim_loop(n_steps: intc, state: ArrF8_2d, model: ArrF8_1d):
            m_sample(0, state)
            for step in range(1, n_steps + 1):
                cpl = c_couple(step)
                new_state = i_scheme(state, model, cpl)
                c_update(step, new_state)
                state = new_state
                m_sample(step, state)

        _sim_loop(n_steps,
                  state,
                  self.model.data)