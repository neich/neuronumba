import numpy as np
from scipy import linalg

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables.base_observable import Observable
from neuronumba.tools.matlab_tricks import correlation_from_covariance, lyap

class LinearFC(Observable):
    A = Attr(default=None)
    Qn = Attr(default=None)
    Vars = Attr(default=2)  # 2 to keep backwards compatibility
    lyap_method = Attr(default='slycot') # Current methods are ‘slycot’ and ‘scipy’.

    def from_matrix(self, A, Qn, Vars=2):  # 2 to keep backwards compatibility
        self.A = A
        self.Qn = Qn
        self.Vars = Vars

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
        Vars: number of variables integrated in the model

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

        # Solves the Lyapunov equation: A*X + X*Ah = Q, with Ah the conjugate transpose of A
        CVth = lyap(self.A, self.Qn, method=self.lyap_method)

        # simulated FC
        FCth = correlation_from_covariance(CVth)
        # Functional connectivity matrix (FC)
        N = int(self.A.shape[0] / self.Vars)
        FC = FCth[0:N, 0:N]
        CV = CVth[0:N, 0:N]

        return {'FC': FC, 'CVth': CVth, 'CV': CV}