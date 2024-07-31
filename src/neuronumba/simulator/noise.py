import numpy as np

from neuronumba.basic.attr import HasAttr, Attr


class Noise(HasAttr):
    pass


class Additive(Noise):

    sigmas = Attr(required=True)
    n_state_vars = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self._n_state_vars = self.sigmas.shape[0]

    def generate(self, n_rois):
        """

        :param n_rois: number of regions of interest
        :return:
        """
        noise = np.zeros((self.n_state_vars, n_rois))
        for i in range(self.n_state_vars):
            if self.sigmas[i] > 0.0:
                noise[i, :] = np.random.normal(0.0, self.sigmas[i], n_rois)
        return noise
