import numpy as np

from neuronumba.basic.attr import HasAttr, Attr


class Connectivity(HasAttr):

    weights = Attr(default=None, required=True)


