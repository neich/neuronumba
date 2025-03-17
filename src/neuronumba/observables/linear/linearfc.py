import numpy as np
from scipy import linalg

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables.base_observable import Observable
from neuronumba.tools.matlab_tricks import correlation_from_covariance, lyap

class LinearFC(Observable):
    A = Attr(default=None)
    Qn = Attr(default=None)
    lyap_method = Attr(default='slycot') # Current methods are ‘slycot’ and ‘scipy’.

    def from_matrix(self, A, Qn):
        self.A = A
        self.Qn = Qn

        return self.compute()

    def _compute(self):
        """
        This function computes the FC from a linearised model.
        solves the equation for the covariances C
                  A Cv + Cv At + Qn = 0

        Parameters
        ----------
        A : (generative) SC, format (n_roi, n_roi)
        Qn: noise matrix, format (n_roi, n_roi)

        Returns
        -------
        FC : functional connectivity matrix, format (n_roi, n_roi)
        CV : time-lagged covariance, format (n_roi, n_roi)
        Cvth : TYPE
            DESCRIPTION.
        """

        if self.A is None or not (isinstance(self.A, np.ndarray) and self.A.ndim == 2):
            raise TypeError("Invalid attribute A")
        if self.Qn is None or not (isinstance(self.Qn, np.ndarray) and self.Qn.ndim == 2):
            raise TypeError("Invalid attribute Qn")

        N = int(self.A.shape[0] / 2)
        # Solves the Lyapunov equation: A*X + X*Ah = Q, with Ah the conjugate transpose of A
        CVth = lyap(self.A, self.Qn, method=self.lyap_method)

        # simulated FC
        FCth = correlation_from_covariance(CVth)
        # Functional connectivity matrix (FC)
        FC = FCth[0:N, 0:N]
        CV = CVth[0:N, 0:N]

        return {'FC': FC, 'CVth': CVth, 'CV': CV}