from numba import float64, njit
import numpy as np

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools.types import ArrF8_2d, ArrF8_1d
from neuronumba.simulator.integrators.base_integrator import Integrator


class EulerDeterministic(Integrator):

    def get_numba_scheme(self, dfun):
        dt = self.dt
        stimulus = np.empty((1, 1))

        @njit(float64[:, :](float64[:, :], float64[:], float64[:, :]))
        def scheme(state: ArrF8_2d, model: ArrF8_1d, coupling: ArrF8_2d):
            d_state = dfun(model, state, coupling)
            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus
            next_state = state + dt * d_state
            return next_state

        return scheme


class EulerStochastic(Integrator):

    sigmas = Attr(default=None, required=True)
    _sqrt_dt = Attr(dependant=True)

    def _init_dependant(self):
        self._sqrt_dt = np.sqrt(self.dt)

    def get_numba_scheme(self, dfun):
        dt = self.dt
        stimulus = np.empty((1, 1))
        sigmas = self.sigmas
        sqrt_dt = self._sqrt_dt

        @njit(float64[:, :](float64[:, :], float64[:], float64[:, :]))
        def scheme(state: ArrF8_2d, model: ArrF8_1d, coupling: ArrF8_2d):
            d_state = dfun(model, state, coupling)
            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus
            noise = np.zeros(state.shape)
            n_rois = state.shape[1]
            for i in range(sigmas.shape[0]):
                if sigmas[i] > 0.0:
                    noise[i] = np.random.normal(loc=0.0, scale=sigmas[i], size=n_rois)

            next_state = state + dt * d_state + sqrt_dt * noise
            return next_state

        return scheme