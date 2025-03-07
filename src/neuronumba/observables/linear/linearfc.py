import numpy as np
from scipy import linalg

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables.base_observable import Observable
from neuronumba.tools.matlab_tricks import correlation_from_covariance

# Necessary if using the "slycot" versions of the solver
# import control
# import numpy as np

class LinearFC(Observable):
    A = Attr(default=None)
    Qn = Attr(default=None)

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
        CVth = linalg.solve_continuous_lyapunov(self.A, -self.Qn)

        # There are two other options to compute the solution to the equation, using the same slycot library under matlab implementation.
        # A) Using the lyap "sylvester" version. Under the hood is calling slycot function "sb04md"
        # Aconjtrans = np.atleast_2d(self.A).T.conj()
        # CVth = control.lyap(self.A, Aconjtrans, self.Qn, method='slycot')
        # B) Using the lyap "lyapunov" version. Under the hood is calling slycot function "sb03md"
        # CVth = control.lyap(self.A, self.Qn, method='slycot')

        # simulated FC
        FCth = correlation_from_covariance(CVth)
        # Functional connectivity matrix (FC)
        FC = FCth[0:N, 0:N]
        CV = CVth[0:N, 0:N]

        return {'FC': FC, 'CVth': CVth, 'CV': CV}