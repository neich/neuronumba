import numpy as np
import numba as nb

from src.neuronumba import HasAttr, Attr
from src.neuronumba import ArrF8_2d, ArrF8_1d
from src.neuronumba import Integrator


class EulerDeterministic(Integrator):

    def get_numba_scheme(self, dfun):
        dt = self.dt
        stimulus = np.empty((1, 1))

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def scheme(state: ArrF8_2d, coupling: ArrF8_2d):
            d_state, observed = dfun(state, coupling)
            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus
            next_state = state + dt * d_state
            return next_state, observed

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

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def scheme(state: ArrF8_2d, coupling: ArrF8_2d):
            d_state, observed = dfun(state, coupling)
            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus
            noise = np.zeros(state.shape)
            n_rois = state.shape[1]
            for i in range(sigmas.shape[0]):
                if sigmas[i] > 0.0:
                    noise[i] = np.random.normal(loc=0.0, scale=sigmas[i], size=n_rois)

            next_state = state + dt * d_state + sqrt_dt * noise
            return next_state, observed

        return scheme