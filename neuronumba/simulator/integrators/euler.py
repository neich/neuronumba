from numba import float64, njit
from numba.experimental import jitclass
import numpy as np
import numpy.typing as npt

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools.types import ArrF8_2d, ArrF8_1d
import neuronumba.simulator.noise as nn_noise


@jitclass
class EulerMaruyama(object):
    dt: float64

    def __init__(self, dt=0.1, noise=nn_noise.Additive):
        self.dt = dt
        self.noise = noise

    def scheme(self, state: npt.ArrayLike, model, coupling, stimulus=None) -> np.ndarray:
        """

        :param state: state variables, shape (n_state__variables, n_rois)
        :param dfun:
        :param coupling:
        :param local_coupling:
        :param stimulus:
        :return:
        """
        d_state = model.dfun(state, coupling)
        noise = self.noise.generate(state.shape[0])
        next_state = state + self.dt * (d_state + stimulus) + noise
        return next_state


class EulerDeterministic(HasAttr):

    dt = Attr(default=0.1, required=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_numba_scheme(self, dfun):
        dt = self.dt
        stimulus = np.empty((1, 1))

        @njit(float64[:,:](float64[:,:], float64[:], float64[:,:]))
        def scheme(state: ArrF8_2d, model: ArrF8_1d, coupling: ArrF8_2d):
            d_state = dfun(model, state, coupling)
            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus
            next_state = state + dt * d_state
            return next_state

        return scheme


class EulerStochastic(HasAttr):

    dt = Attr(default=None, required=True)
    sigmas = Attr(default=None, required=True)

    def configure(self, **kwargs):
        super().configure(**kwargs)
        self._sqrt_dt = np.sqrt(self.dt)

    def get_numba_scheme(self, dfun):
        dt = self.dt
        stimulus = np.empty((1, 1))
        sigmas = self.sigmas
        sqrt_dt = self._sqrt_dt

        @njit(float64[:,:](float64[:,:], float64[:], float64[:,:]))
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