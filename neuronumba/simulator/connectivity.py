import numpy as np

from neuronumba.basic.attr import HasAttr, Attr


class Connectivity(HasAttr):

    weights = Attr(default=None, required=True)
    lengths = Attr(default=None, required=True)
    speed = Attr(default=1e6, required=False)
    delays = Attr(default=None, required=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delays = self.lengths / self.speed

    @property
    def n_rois(self):
        return self.weights.shape[0]


