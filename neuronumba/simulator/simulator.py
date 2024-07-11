import numpy as np

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.simulator import models, integrators, connectivity


class Simulator(HasAttr):
    connectivity = Attr(
        default=None,
        required=True)

    model = Attr(
        default=None,
        required=True)

    integrator = Attr(
        default=integrators.EulerDeterministic(),
        required=True)

    def run(self, t_start=0, t_end=10000, stimulus=None):
        assert self.connectivity, "No connectivity defined for simulation!"

        dt = self.integrator.dt
        n_steps = (t_end - t_start)
        n_rois = self.connectivity.weights.shape[0]
        state = self.model.initial(n_rois)
        self._state_shape = (int(self.model.n_state_vars), n_rois)
        for step in range(1, n_steps + 1):
            coupling = self._loop_compute_coupling(step)
            if stimulus:
                raise Exception("Stimulus not implemented!")
            state = self.integrator.scheme(state, self.model, coupling)


    def _loop_compute_coupling(self, step):
        return np.zeros(self._state_shape)
