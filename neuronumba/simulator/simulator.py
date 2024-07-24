import numba as nb

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.numba_tools.types import ArrF8_2d, ArrF8_1d


class Simulator(HasAttr):

    connectivity = Attr(required=True)

    model = Attr(required=True)

    integrator = Attr(required=True)

    coupling = Attr(required=True)

    monitors = Attr(required=True)

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
        init_state = self.model.initial_state(n_rois)
        init_observed = self.model.initial_observed(n_rois)
        self._state_shape = (int(self.model.n_state_vars), n_rois)

        for m in self.monitors:
            m.configure(dt=dt, t_max=t_max, n_rois=n_rois)

        c_couple = self.coupling.get_numba_couple()
        c_update = self.coupling.get_numba_update()
        i_scheme = self.integrator.get_numba_scheme(self.model.get_numba_dfun())
        m_sample = self.monitors[0].get_numba_sample()

        c_update(0, init_state)

        @nb.njit(nb.void(nb.intc,  # n_steps
                   nb.f8[:, :],  # initial state variables
                   nb.f8[:, :]  # initial observed variables
                   )
              )
        def _sim_loop(n_steps: nb.intc, state: ArrF8_2d, observed: ArrF8_2d):
            m_sample(0, state, observed)
            for step in range(1, n_steps + 1):
                cpl = c_couple(step)
                new_state, new_observed = i_scheme(state, cpl)
                c_update(step, new_state)
                m_sample(step, new_state, new_observed)
                state = new_state

        _sim_loop(n_steps, init_state, init_observed)