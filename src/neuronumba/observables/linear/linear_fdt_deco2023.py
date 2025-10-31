# =======================================================================
# Computes the Viiolation of the Fluctuation-Dissipation Theorem.
# 
# From:
# Gustavo Deco et al. ,Violations of the fluctuation-dissipation theorem 
# reveal distinct nonequilibrium dynamics of brain states.
# Phys. Rev. E108,064410(2023).DOI:10.1103/PhysRevE.108.064410
#
# Derived from matlab's Irene Acero Pousa code and ported by Albert Juncà
# =======================================================================

import numpy as np
from neuronumba.observables.base_observable import Observable
from neuronumba.basic.attr import Attr
from neuronumba.simulator.models import Model
from neuronumba.observables.linear.linearfc import LinearFC

class LinearFdtDeco2023(Observable):
    A = Attr(default=None)
    Qn = Attr(default=None)
    sigma = Attr(default=0.01)

    def from_matrix(self, A, Qn):
        self.A = A
        self.Qn = Qn

        return self.compute()

    def _compute(self):

        if self.A is None or not (isinstance(self.A, np.ndarray) and self.A.ndim == 2):
            raise TypeError("Invalid attribute A")
        if self.Qn is None or not (isinstance(self.Qn, np.ndarray) and self.Qn.ndim == 2):
            raise TypeError("Invalid attribute Qn")
        
        n_roi = int(self.A.shape[0] / 2)
        # n2 = 2 * n_roi
        
        obs = LinearFC()
        result =  obs.from_matrix(self.A, self.Qn)
        # FC_sim = result['FC']
        COVsimtotal = result['CVth']
        # COV_sim = result['CV']
        
        # Inverse of the Jacobian Matrix
        invA = np.linalg.inv(self.A)

        # The following two lines are the linearized code from:
        #
        # chi = np.zeros((n2, n2))
        # chi2 = np.zeros((n2, n2))
        # for i in range(n2):
        #     for j in range(n2):
        #         # vector with all components equal to 0 exccelt the j component, 
        #         # which equals the value of the perturbation
        #         hh = np.zeros((n2, 1))
        #         hh[j, 0] = 1
        #         # Compute the perturbation response: xepsilon = -invA * hh.
        #         # (Since hh is the j-th basis vector, this is just the j-th column of -invA.)
        #         xepsilon = -np.dot(invA, hh)
        #         # Compute the deviation (chi) and its absolute value (chi2)
        #         chi[i, j]  = np.abs((2 * COVsimtotal[i, j] / sigma**2) - xepsilon[i, 0])
        #         chi2[i, j] = np.abs(xepsilon[i, 0])
        #
        chi = np.abs((2 * COVsimtotal / self.sigma**2) + invA)
        chi2 = np.abs(invA)

        # average over regions
        chi_j = np.mean(chi[:n_roi, :n_roi], axis=0) / np.mean(chi2[:n_roi, :n_roi], axis=0)
        # level of non-equilibrium
        FDTm = np.mean(chi_j)

        return {
            'regions_fdt': chi_j,
            'global_fdt': FDTm
        }


