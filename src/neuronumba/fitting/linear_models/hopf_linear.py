# -*- coding: utf-8 -*-
# =======================================================================
# Linear approximation to the Hopf model
# from
# Ponce-Alvarez, A., Deco, G. The Hopf whole-brain model and its linear approximation.
# Sci Rep 14, 2615 (2024).
# https://doi.org/10.1038/s41598-024-53105-0
#
# @author: Eider, modified by Wiep, modified by dagush
# Goal: Create functions for the linearized Hopf model
# =======================================================================

# import necessary packages
import numpy as np
from scipy import linalg

from neuronumba.fitting.linear_models.base_linear import BaseLinearModel
from neuronumba.tools.matlab_tricks import correlation_from_covariance


class HopfLinearModel(BaseLinearModel):

    def compute_matrix(self, sc, sigma):
        """
        This function computes the linearised Hopf model.
        solves the equation for the covariances C
                  A Cv + Cv At + Qn = 0
        
        Parameters
        ----------
        sc : (generative) SC, format (n_roi, n_roi)
        sigma : ..., format one value, type float
        
        Returns
        -------
        FC : functional connectivity matrix, format (n_roi, n_roi)
        CV : time-lagged covariance, format (n_roi, n_roi)
        Cvth : TYPE
            DESCRIPTION.
        A : TYPE
            DESCRIPTION.
        """
        # =============== Specific Hopf computations
        N = len(sc)  # number of nodes
        wo = np.atleast_2d(self.model.omega).T.conj() * (2 * np.pi)  # intrinsic node frequency
        
        # --------------- Jacobian
        s = np.sum(sc, axis=1)  #  vector containing the strength of each node
        B = np.diag(s)  # create a zero matrix with "s" in the diagonal
        
        Axx = self.model.a * np.eye(N) - B + sc  # Axx = Ayy = diag(a-s) + sc, diagonal entries of the Jacobian matrix
        Ayy = Axx
        Axy = -np.diag(wo[:,0])       # Axy = -Ayx = diag(w)
        Ayx = np.diag(wo[:,0])
        A = np.block([[Axx, Axy], [Ayx, Ayy]])  # create Jacobian matrix
        
        # =============== Build Qn
        Qn = (sigma**2) * np.eye(2*N)  # covariance matrix of the noise
        # Aconjtrans = np.atleast_2d(A).T.conj()  # A.T -> not needed for Lyapunov
        
        # =============== Solve Sylvester equation and compute observable
        # Cvth = linalg.solve_sylvester(A, Aconjtrans, -Qn)  # solves the Sylvestser equation: A*X + X*B = Q  => A = A, B = At, Q = -Qn
        Cvth= linalg.solve_continuous_lyapunov(A, -Qn)  # Solves the Lyapunov equation: A*X + X*Ah = Q, with Ah the conjugate transpose of A
        FCth = correlation_from_covariance(Cvth)  # simulated FC
        FC = FCth[0:N, 0:N]  # Functional connectivity matrix (FC)
        CV = Cvth[0:N, 0:N]  # Covariance matrix (CV)
        
        return FC, CV, Cvth, A