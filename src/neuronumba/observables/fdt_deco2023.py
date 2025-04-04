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

class FdtDeco2023(Observable):
    sigma = Attr(default=0.01)
    eff_con = Attr(default=None)
    model = Attr(default=None)

    def _compute(self):
        if self.eff_con is None or not (isinstance(self.eff_con, np.ndarray) and self.eff_con.ndim == 2):
            raise TypeError("Invalid effective connectivity")

        if self.model is None or not isinstance(self.model, Model):
            raise TypeError("Invalid model")
        
        n_roi = np.shape(self.eff_con)[0]
        n2 = 2 * n_roi
        
        A = self.model.get_jacobian(self.eff_con)
        Qn = self.model.get_noise_matrix(self.sigma, len(self.eff_con))
        obs = LinearFC()
        result =  obs.from_matrix(A, Qn)
        FC_sim = result['FC']
        COVsimtotal = result['CVth']
        COV_sim = result['CV']

        # Inverse of the Jacobian Matrix
        invA = np.linalg.inv(A)

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
        chij = np.mean(chi[:n_roi, :n_roi], axis=0) / np.mean(chi2[:n_roi, :n_roi], axis=0)
        # level of non-equilibrium
        FDTm = np.mean(chij)

        return {
            'regions_fdt': chij,
            'global_fdt': FDTm
        }


