from scipy import linalg

from neuronumba.observables.linear.base_linear import ObservableLinear
from neuronumba.tools.matlab_tricks import correlation_from_covariance

# Necessary if using the "slycot" versions of the solver
# import control
# import numpy as np

class LinearFC(ObservableLinear):
    def from_matrix(self, A, Qn):
        N = int(A.shape[0] / 2)
        # Solves the Lyapunov equation: A*X + X*Ah = Q, with Ah the conjugate transpose of A
        CVth = linalg.solve_continuous_lyapunov(A, -Qn)

        # There are two other options to compute the solution to the equation, using the same slycot library under matlab implementation.
        # A) Using the lyap "sylvester" version. Under the hood is calling slycot function "sb04md"
        # Aconjtrans = np.atleast_2d(A).T.conj()
        # CVth = control.lyap(A, Aconjtrans, Qn, method='slycot')
        # B) Using the lyap "lyapunov" version. Under the hood is calling slycot function "sb03md"
        # CVth = control.lyap(A, Qn, method='slycot')

        # simulated FC
        FCth = correlation_from_covariance(CVth)
        # Functional connectivity matrix (FC)
        FC = FCth[0:N, 0:N]
        CV = CVth[0:N, 0:N]

        return {'FC': FC, 'CVth': CVth, 'CV': CV}