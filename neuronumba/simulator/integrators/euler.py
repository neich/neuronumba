from numba import float64
from numba.experimental import jitclass
import numpy as np
import numpy.typing as npt

from neuronumba.simulator.integrators.base_integrator import Integrator__init__
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
        d_state = model.dfun(state,coupling)
        noise = self.noise.generate(state.shape[0])
        next_state = state + self.dt * (d_state + stimulus) + noise
        return next_state


@jitclass
class EulerDeterministic(object):
    dt: float64

    def __init__(self, dt=0.1):
        self.dt = dt

    def scheme(self, state: npt.ArrayLike, model, coupling, stimulus=None) -> np.ndarray:
        """

        :param state: state variables, shape (n_state__variables, n_rois)
        :param dfun:
        :param coupling:
        :param local_coupling:
        :param stimulus:
        :return:
        """
        d_state = model.dfun(state,coupling)
        next_state = state + self.dt * (d_state + stimulus)
        return next_state
