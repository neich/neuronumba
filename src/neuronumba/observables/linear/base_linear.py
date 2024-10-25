import numpy as np
from neuronumba.basic.attr import HasAttr, Attr


class ObservableLinear(HasAttr):
    """
    Abstract class for Observables based on linear matrices.

    NOTES: Implementation is as this to maximize the portability with the old class based library.

    """

    def from_matrix(self, A, Qn):
        """ Main method to compute the Observable from an fMRI BOLD signal.

        Args:
            A: model linear matrix
            Qn: noise covariance matrix
        Returns:
            dict: dictionary with the results
        """

        raise NotImplemented('Should have been implemented by subclass!')