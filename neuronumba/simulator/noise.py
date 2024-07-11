from numba.experimental import jitclass
import numpy as np


class Noise(object):

    def __init__(self):
        pass


@jitclass
class Additive(Noise):

    def __init__(self, sigmas):
        """

        :param sigmas: std deviation from normal distribution for noise generation.
            Shape is (number_of_state_variables,)
        """
        super().__init__()
        self.sigmas = sigmas
        self._n_state_vars = sigmas.shape[0]

    def generate(self, n_rois):
        """

        :param n_rois: number of regions of interest
        :return:
        """
        noise = np.zeros((self._n_state_vars, n_rois))
        for i in range(self._n_state_vars):
            if self.sigmas[i] > 0.0:
                noise[i, :] = np.random.normal(0.0, self.sigmas[i], n_rois)
        return noise
