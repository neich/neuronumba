from neuronumba.basic.attr import HasAttr, Attr


class Connectivity(HasAttr):

    weights = Attr(default=None, required=True)
    lengths = Attr(default=None, required=True)
    speed = Attr(default=1e6, required=False)

    def _init_dependant(self):
        self.n_rois = self.weights.shape[0]


