# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
#
# For the linear version, we use [Deco_2014] and
# [Demirtaş_2019] M. Demirtaş, J.B. Burt, M. Helmer, J.L. Ji, B.D. Adkinson, M.F. Glasser,
#                 D.C. Van Essen, S.N. Sotiropoulos, A. Anticevic, J.D. Murray
#                 Hierarchical Heterogeneity across Human Cortex Shapes Large-Scale Neural Dynamics
#                 Volume 101, Issue 6, p1181-1194.e13, March 20, 2019
#
# ==========================================================================
# ==========================================================================
import numpy as np
import numba as nb
from scipy.optimize import fsolve
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.fitting.fic.fic import FICHerzog2022
from neuronumba.numba_tools.addr import address_as_void_pointer
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model
from neuronumba.simulator.models import LinearCouplingModel


class Deco2014(LinearCouplingModel):
    # Se, excitatory synaptic activity
    state_vars = Model._build_var_dict(['S_e', 'S_i'])
    n_state_vars = len(state_vars)
    c_vars = [0]

    # Ie, excitatory current
    # re, excitatory firing rate
    observable_vars = Model._build_var_dict(['Ie', 're'])
    n_observable_vars = len(observable_vars)

    auto_fic = Attr(default=False, attributes=Model.Type.Model)
    tao_e = Attr(default=100.0, attributes=Model.Type.Model)
    tao_i = Attr(default=10.0, attributes=Model.Type.Model)
    gamma_e = Attr(default=0.641, attributes=Model.Type.Model)
    gamma_i = Attr(default=1.0, attributes=Model.Type.Model)
    I0 = Attr(default=0.382, attributes=Model.Type.Model)     # [nA] overall effective external input
    w = Attr(default=1.4, attributes=Model.Type.Model)
    J_NMDA = Attr(default=0.15, attributes=Model.Type.Model)  # [nA] NMDA current
    Jext_e = Attr(default=1.0, attributes=Model.Type.Model)
    Jext_i = Attr(default=0.7, attributes=Model.Type.Model)
    ae = Attr(default=310.0, attributes=Model.Type.Model)
    be = Attr(default=125.0, attributes=Model.Type.Model)
    de = Attr(default=0.16, attributes=Model.Type.Model)
    ai = Attr(default=615.0, attributes=Model.Type.Model)
    bi = Attr(default=177.0, attributes=Model.Type.Model)
    di = Attr(default=0.087, attributes=Model.Type.Model)
    J = Attr(default=1.0, attributes=Model.Type.Model)
    I_external = Attr(default=0.0, attributes=Model.Type.Model)

    recompute_steady_state = Attr(default=False, attributes=Model.Type.Model)

    @overrides
    def _init_dependant(self):
        super()._init_dependant()
        if self.auto_fic and not self._attr_defined('J'):
            self.J = FICHerzog2022().compute_J(self.weights, self.g)

    @property
    def get_state_vars(self):
        return Deco2014.state_vars

    @property
    def get_observablevars(self):
        return Deco2014.observable_vars

    @property
    def get_c_vars(self):
        return Deco2014.c_vars

    def initial_state(self, n_rois):
        state = np.empty((Deco2014.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        return state

    def initial_observed(self, n_rois):
        observed = np.empty((Deco2014.n_observable_vars, n_rois))
        observed[0] = 0.0
        observed[1] = 0.0
        return observed

    def get_numba_dfun(self):
        m = self.m.copy()
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Deco2014_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            # Clamping, needed in Deco2014 model and derivatives...
            Se = state[0, :].clip(0.0,1.0)
            Si = state[1, :].clip(0.0,1.0)

            # Eq for I^E (5). I_external = 0 => resting state condition.
            Ie = m[np.intp(P.Jext_e)] * m[np.intp(P.I0)] + m[np.intp(P.w)] * m[np.intp(P.J_NMDA)] * Se + m[np.intp(P.J_NMDA)] * coupling[0, :] - m[np.intp(P.J)] * Si + m[np.intp(P.I_external)]
            # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
            Ii = m[np.intp(P.Jext_i)] * m[np.intp(P.I0)] + m[np.intp(P.J_NMDA)] * Se - Si
            y = m[np.intp(P.ae)] * Ie - m[np.intp(P.be)]
            # In the paper re was g_E * (I^{(E)_n} - I^{(E)_{thr}}). In the paper (7)
            # Here, we distribute as g_E * I^{(E)_n} - g_E * I^{(E)_{thr}}, thus...
            re = y / (1.0 - np.exp(-m[np.intp(P.de)] * y))
            y = m[np.intp(P.ai)] * Ii - m[np.intp(P.bi)]
            # In the paper ri was g_I * (I^{(I)_n} - I^{(I)_{thr}}). In the paper (8)
            # Apply same distributing as above...
            ri = y / (1.0 - np.exp(-m[np.intp(P.di)] * y))
            # divide by 1000 because we need milliseconds!
            dSe = -Se / m[np.intp(P.tao_e)] + m[np.intp(P.gamma_e)] * (1. - Se) * re / 1000.
            dSi = -Si / m[np.intp(P.tao_i)] + m[np.intp(P.gamma_i)] * ri / 1000.
            return np.stack((dSe, dSi)), np.stack((Ie, re))

        return Deco2014_dfun

    # ==========================================================================
    # Computes the steady state of the DMF
    # Network node model: used for the fsolve to find the fixed point
    #      Ie : N np array - Synaptic E currents
    #      Ii : N np array - Synaptic I currents
    #      re : N np array - E firing rates
    #      ri : N np array - I firing rates
    #      Se : N np array - E synaptic gating
    #      Si : N np array - I synaptic gating
    #      J  : N np array - Inhibitory coupling strengths (of self-inhibition)
    def _compute_steady_state(self, SC):
        N = len(SC)
        # Parameters  ------------------------------------------------------------------
        # Steady-state solutions in isolated case (values from [Demirtaş_2019]
        # -- Excitatory
        re_ss = 3.0773270642  # Hz
        Ie_ss = 0.3773805650  # nA
        Se_ss = 0.1647572075  # dimensionless
        # -- Inhibitory
        ri_ss = 3.9218448633  # Hz
        Ii_ss = 0.2528951325  # nA
        Si_ss = 0.0392184486  # dimensionless

        # ==========================================================================
        # eye = np.identity(N)
        # K_EE = (self.w * eye) + (self.J_NMDA * SC)
        # K_EI = (w_EI * eye)
        if self.recompute_steady_state:
            # Excitatory and inhibitory connection weights

            # Recompute the steady state for Ie
            # (7) -> y = ae * Ie - be
            #        re = y / (1.0 - np.exp(-de * y))
            # Thus
            #        0 = y / (1.0 - np.exp(-de * y)) - re
            phi_e_0 = lambda Ie : (self.ae * Ie - self.be) / (1.0 - np.exp(-self.de * (self.ae * Ie - self.be))) - re_ss
            Ie_ss, infodict, ier, mesg = fsolve(phi_e_0, x0=Ie_ss, full_output=True)

            # Compute the steady state for Se.
            # Use Equation (9) in the paper, but for steady state dSe = 0
            #        dSe = 0 = -Se / tao_e + gamma_e * (1. - Se) * re
            # Thus we get:
            #        Se = gamma_e * tao_e * re / (1 + gamma_e * tao_e * re)
            Se_ss = self.gamma_e * self.tao_e * re_ss/1000. / (1 + self.gamma_e * self.tao_e * re_ss/1000.)

            # Local feedback inhbition
            # phi_i = lambda Ii: (self.ai * Ii - self.bi) / (1.0 - np.exp(-self.di * (self.ai * Ii - self.bi)))
            # Numerically solve for inhibitory currents first:
            #   (6)  -> Ii = Jext_i * I0 + J_NMDA * Se - Si
            #   (10) -> dSi = 0 = -Si / tao_i + gamma_i * ri
            #   (8)  -> ri = phi_i(Ii)
            # Thus:
            #    0 = Jext_i * I0 + J_NMDA * Se - gamma_i * tao_i * phi_i(Ii) - Ii
            # inh_curr = lambda I : self.Jext_i * self.I0 + self.J_NMDA * Se_ss - \
            #                       self.gamma_i * self.tao_i * phi_i(I) - I
            # Ii_ss, infodict, ier, mesg = fsolve(inh_curr, x0=Ii_ss, full_output=True)
            # # Assuming no error, compute ri_ss and Si_ss
            # ri_ss = phi_i(Ii_ss)  # compute new steady state rate
            # Si_ss = ri_ss * self.tao_i * self.gamma_i  # update stored val.

        # Solve for J using the steady state values (fixed points)
        # Eq for I^E (5)
        #    Ie = Jext_e * I0 + w * J_NMDA * Se + J_NMDA * SC@Se - J * Si + I_external
        # Thus
        #     J = (Jext_e * I0 + w * J_NMDA * Se + J_NMDA * SC@Se + I_external - Ie) / Si
        J = (self.Jext_e * self.I0 +
             self.w * self.J_NMDA * Se_ss +
             self.J_NMDA * SC@(np.ones(N)*Se_ss) +
             self.I_external - Ie_ss) / Si_ss

        return Ie_ss, Ii_ss, re_ss, ri_ss, Se_ss, Si_ss, J

    def get_jacobian(self, SC):
        """
         This function returns the analytical solution for the Jacobian of the DMF
         with respect to SN and SG in the network case.

        Parameters
        ----------
        sc : (generative) SC, format (n_roi, n_roi) Note: if we use a global coupling G, it should have been pre-multiplied to sc

        Returns
        -------
        J : Matrix (NxN)
            The Jacobian Matrix.
        Qn : Matrix (NxN)
            The covariance noise matrix

        Notes
        -----
        This method should be executed before calculating the linearized covariance or performing numerical integration,
        each time the model parameters are modified.
        """
        # Parameters  ------------------------------------------------------------------
        N = len(SC)

        # Biophysical Parameters
        taoE = self.tao_e  # params["taoE"]
        taoI = self.tao_i  # params["taoI"]
        gammaE = self.gamma_e  # params["gammaE"]
        JN = self.J_NMDA  # params["JN"]
        I0 = self.I0  # params["I0"]
        wE = self.Jext_e  # params["wE"]
        wI = self.Jext_i  # params["wI"]
        # gain = params["gain"]
        w_EE = self.w  # params["w_EE"]
        # w_EI = self.J_NMDA  #params["w_EI"]
        g_i = self.di  # params["g_i"]
        b_i = self.bi  # params["I_i"]
        c_i = self.ai  # params["c_i"]
        c_e = self.ae  # params["c_e"]
        b_e = self.be  # params["I_e"]
        g_e = self.de  # params["g_e"]

        # compute_steady_state
        Ie, Ii, re, ri, Se, Si, J = self._compute_steady_state(SC)
        Ie = Ie * np.ones(N); Ii = Ii * np.ones(N);
        re = re * np.ones(N); ri = ri * np.ones(N);
        Se = Se * np.ones(N); Si = Si * np.ones(N)

        # Initialize Jacobian ----------------------------------------------------
        jacobian = np.zeros([N * 2, N * 2])
        first_q_diag_indices = (np.arange(0, N), np.arange(0, N))  # Generate diagonal indices
        second_q_diag_indices = (np.arange(0, N), np.arange(N, N + N))
        third_q_diag_indices = (np.arange(N, N + N), np.arange(0, N))
        fourth_q_diag_indices = (np.arange(N, N + N), np.arange(N, N + N))

        # ----- Find partial derivatvies-------------------------------------------
        # -- derivatives of SiE with respect to all other variables ---------------
        U = (c_e * Ie - b_e)
        U_star = g_e * U
        M = ((c_e * (1 - np.exp(-U_star))) - (U * c_e * g_e * np.exp(-U_star))) / ((1 - np.exp(-U_star)) ** 2)
        # dSiE_dSiE: This should be an N dimensional vector // rn = riE & sn = SiE
        dSiE_dSiE = (-1 / taoE) + gammaE * (-re + (1 - Se) * M * w_EE * JN) / 1000  # These are the diagonal entris of the first quadrant
        jacobian[first_q_diag_indices] = dSiE_dSiE
        # Note: Important to divide all r values by 1000 as the system runs on ms!!
        # dSiE_dSjE: This fills the non-diagonal entries of J first quarant
        for i in range(N):
            for j in range(N):
                if i != j:
                    dSiE_dSjE = (1 - Se[i]) * gammaE * M[i] * JN * SC[i, j] / 1000
                    jacobian[i, j] = dSiE_dSjE
        # dSiE_dSiI: This should be an N dimensional vector // rn = riE & sn = SiE
        dSiE_dSiI = (1 - Se) * gammaE * M * (-J) / 1000  # These are the diagonal entries of the first quadrant
        jacobian[second_q_diag_indices] = dSiE_dSiI

        # -- derivatives of SiI with respect to all other variables ---------------
        U = (c_i * Ii - b_i)
        U_star = g_i * U
        M_2 = ((c_i * (1 - np.exp(-U_star))) - (U * g_i * c_i * np.exp(-U_star))) / ((1 - np.exp(-U_star)) ** 2)
        # dSiI_dSiE: This should be an N dimensional vector
        dSiI_dSiE = M_2 * JN / 1000  # These are the diagonal entries of the third quadrant
        jacobian[third_q_diag_indices] = dSiI_dSiE
        # dSiI_dSiI: This should be an N dimensional vector
        dSiI_dSiI = (-1 / taoI) - M_2 / 1000  # These are the diagonal entries of the last quadrant
        jacobian[fourth_q_diag_indices] = dSiI_dSiI

        # K_IE = - w_IE * eye
        # K_II = - w_II * eye
        #
        # # 4. Derivatives of transfer function for each cell type
        # # at steady state value of current
        # dr_E = self.dphi_E(I_E_ss) * eye
        # dr_I = self.dphi_I(I_I_ss) * eye
        #
        # # A_{mn} = dS_i^m/dS_j^n
        # A_EE = (-1. / tau_E - (gamma * r_E_ss)) * eye + \
        #        ((-gamma * (S_E_ss - 1.)) * eye).dot(dr_E.dot(K_EE))
        #
        # A_IE = ((gamma * (1. - S_E_ss)) * eye).dot(dr_E.dot(K_IE))
        # A_EI = gamma_I * dr_I.dot(K_EI)
        # A_II = (-1. / tau_I) * eye + gamma_I * dr_I.dot(K_II)
        #
        # # Stack blocks to form full Jacobian
        # jacobian = np.hstack((np.vstack((A_EE, A_EI)),
        #                       np.vstack((A_IE, A_II))))

        return jacobian
