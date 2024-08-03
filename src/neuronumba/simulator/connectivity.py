from neuronumba.basic.attr import HasAttr, Attr


class Connectivity(HasAttr):

    # This is the connectivity matrix. A row specifies the INCOMING weights to a given regions of interest
    weights = Attr(default=None, required=True)
    lengths = Attr(default=None, required=True)
    speed = Attr(default=1e6, required=False)

    n_rois = Attr(dependant=True)

    def _init_dependant(self):
        self.n_rois = self.weights.shape[0]


