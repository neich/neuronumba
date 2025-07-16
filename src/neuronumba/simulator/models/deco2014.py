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
from typing import Dict, List, Tuple, Union, Optional
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
    """
    Deco2014 Dynamic Mean Field (DMF) Model.
    
    This class implements the reduced Wong-Wang neural mass model as described in 
    Deco et al. (2014). The model describes the dynamics of excitatory and inhibitory
    neural populations through synaptic gating variables (S_e, S_i) and includes
    observable variables for excitatory current (Ie) and firing rate (re).
    
    The model equations are:
    - dS_e/dt = -S_e/tau_e + gamma_e * (1 - S_e) * r_e / 1000
    - dS_i/dt = -S_i/tau_i + gamma_i * r_i / 1000
    
    Where firing rates are computed using sigmoid activation functions:
    - r_e = (a_e * I_e - b_e) / (1 - exp(-d_e * (a_e * I_e - b_e)))
    - r_i = (a_i * I_i - b_i) / (1 - exp(-d_i * (a_i * I_i - b_i)))
    
    State Variables:
        S_e: Excitatory synaptic gating variable (dimensionless, 0-1)
        S_i: Inhibitory synaptic gating variable (dimensionless, 0-1)
        
    Observable Variables:
        Ie: Excitatory current (nA)
        re: Excitatory firing rate (Hz)
        
    Coupling Variables:
        S_e: Only excitatory synaptic activity contributes to coupling
    """
    
    # State variables: S_e (excitatory synaptic activity), S_i (inhibitory synaptic activity)
    state_vars = Model._build_var_dict(['S_e', 'S_i'])
    n_state_vars = len(state_vars)
    c_vars = [0]  # Only S_e couples between regions

    # Observable variables: Ie (excitatory current), re (excitatory firing rate)
    observable_vars = Model._build_var_dict(['Ie', 're'])
    n_observable_vars = len(observable_vars)

    # ==========================================================================
    # Model Parameters
    # ==========================================================================
    
    # Automatic FIC computation
    auto_fic = Attr(default=False, attributes=Model.Type.Model,
                   doc="Whether to automatically compute inhibitory coupling strength J using FIC")
    
    # Time constants (ms)
    tau_e = Attr(default=100.0, attributes=Model.Type.Model,
                doc="Excitatory population time constant (ms)")
    tau_i = Attr(default=10.0, attributes=Model.Type.Model,
                doc="Inhibitory population time constant (ms)")
    
    # Synaptic efficacy parameters
    gamma_e = Attr(default=0.000641, attributes=Model.Type.Model,
                  doc="Excitatory synaptic efficacy")
    gamma_i = Attr(default=0.001, attributes=Model.Type.Model,
                  doc="Inhibitory synaptic efficacy")
    
    # External input parameters
    I0 = Attr(default=0.382, attributes=Model.Type.Model,
             doc="Overall effective external input (nA)")
    Jext_e = Attr(default=1.0, attributes=Model.Type.Model,
                 doc="External input scaling for excitatory population")
    Jext_i = Attr(default=0.7, attributes=Model.Type.Model,
                 doc="External input scaling for inhibitory population")
    
    # Recurrent connectivity parameters
    w = Attr(default=1.4, attributes=Model.Type.Model,
            doc="Local recurrent excitatory connection strength")
    J_NMDA = Attr(default=0.15, attributes=Model.Type.Model,
                 doc="NMDA synaptic coupling strength (nA)")
    J = Attr(default=1.0, attributes=Model.Type.Model,
            doc="Local inhibitory coupling strength")
    
    # Sigmoid activation function parameters for excitatory population
    ae = Attr(default=310.0, attributes=Model.Type.Model,
             doc="Excitatory gain parameter (nC^-1)")
    be = Attr(default=125.0, attributes=Model.Type.Model,
             doc="Excitatory threshold parameter (Hz)")
    de = Attr(default=0.16, attributes=Model.Type.Model,
             doc="Excitatory slope parameter (s)")
    
    # Sigmoid activation function parameters for inhibitory population
    ai = Attr(default=615.0, attributes=Model.Type.Model,
             doc="Inhibitory gain parameter (nC^-1)")
    bi = Attr(default=177.0, attributes=Model.Type.Model,
             doc="Inhibitory threshold parameter (Hz)")
    di = Attr(default=0.087, attributes=Model.Type.Model,
             doc="Inhibitory slope parameter (s)")
    
    # External stimulation
    I_external = Attr(default=0.0, attributes=Model.Type.Model,
                     doc="Additional external current input (nA)")
    
    # Steady state computation options
    recompute_steady_state = Attr(default=False, attributes=Model.Type.Model,
                                 doc="Whether to recompute steady state values")

    # ==========================================================================
    # Constants for steady state computation (from Demirtaş et al. 2019)
    # ==========================================================================
    _SS_RE = 3.0773270642  # Hz - Steady state excitatory firing rate
    _SS_IE = 0.3773805650  # nA - Steady state excitatory current
    _SS_SE = 0.1647572075  # dimensionless - Steady state excitatory synaptic gating
    _SS_RI = 3.9218448633  # Hz - Steady state inhibitory firing rate  
    _SS_II = 0.2528951325  # nA - Steady state inhibitory current
    _SS_SI = 0.0392184486  # dimensionless - Steady state inhibitory synaptic gating
    
    # Unit conversion constant
    _MS_TO_S = 1000.0  # Conversion factor from Hz to Hz/ms

    @overrides
    def _init_dependant(self):
        super()._init_dependant()
        if self.auto_fic and not self._attr_defined('J'):
            self.J = FICHerzog2022().compute_J(self.weights, self.g)

    @property
    def get_state_vars(self) -> Dict[str, int]:
        """Get dictionary mapping state variable names to their indices."""
        return Deco2014.state_vars

    @property
    def get_observablevars(self) -> Dict[str, int]:
        """Get dictionary mapping observable variable names to their indices."""
        return Deco2014.observable_vars

    @property
    def get_c_vars(self) -> List[int]:
        """Get list of coupling variable indices."""
        return Deco2014.c_vars

    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Initialize state variables for the model.
        
        Args:
            n_rois: Number of regions of interest
            
        Returns:
            Initial state array with shape (n_state_vars, n_rois)
        """
        state = np.empty((Deco2014.n_state_vars, n_rois))
        state[0] = 0.001  # S_e initial value
        state[1] = 0.001  # S_i initial value
        return state

    def initial_observed(self, n_rois: int) -> np.ndarray:
        """
        Initialize observable variables for the model.
        
        Args:
            n_rois: Number of regions of interest
            
        Returns:
            Initial observed array with shape (n_observable_vars, n_rois)
        """
        observed = np.empty((Deco2014.n_observable_vars, n_rois))
        observed[0] = 0.0  # Ie initial value
        observed[1] = 0.0  # re initial value
        return observed

    def get_numba_dfun(self):
        """
        Generate the Numba-compiled differential function for the Deco2014 model.
        
        Returns:
            Compiled function that computes state derivatives and observables
        """
        m = self.m.copy()
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Deco2014_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Compute derivatives of state variables and observable variables.
            
            Args:
                state: Current state array (2, n_rois) containing [S_e, S_i]
                coupling: Coupling input array (1, n_rois) containing coupled S_e
                
            Returns:
                Tuple of (state_derivatives, observables)
                - state_derivatives: (2, n_rois) array with [dS_e/dt, dS_i/dt]
                - observables: (2, n_rois) array with [Ie, re]
            """
            # Clamping synaptic gating variables to [0,1] range
            Se = state[0, :].clip(0.0, 1.0)
            Si = state[1, :].clip(0.0, 1.0)

            # Compute excitatory current I^E (Equation 5 in Deco et al. 2014)
            # I_e = J_ext_e * I_0 + w * J_NMDA * S_e + J_NMDA * coupling - J * S_i + I_external
            Ie = (m[np.intp(P.Jext_e)] * m[np.intp(P.I0)] + 
                  m[np.intp(P.w)] * m[np.intp(P.J_NMDA)] * Se + 
                  m[np.intp(P.J_NMDA)] * coupling[0, :] - 
                  m[np.intp(P.J)] * Si + 
                  m[np.intp(P.I_external)])
            
            # Compute inhibitory current I^I (Equation 6 in Deco et al. 2014)
            # I_i = J_ext_i * I_0 + J_NMDA * S_e - S_i
            Ii = (m[np.intp(P.Jext_i)] * m[np.intp(P.I0)] + 
                  m[np.intp(P.J_NMDA)] * Se - Si)
            
            # Compute excitatory firing rate using sigmoid function (Equation 7)
            # r_e = (a_e * I_e - b_e) / (1 - exp(-d_e * (a_e * I_e - b_e)))
            y_e = m[np.intp(P.ae)] * Ie - m[np.intp(P.be)]
            re = y_e / (1.0 - np.exp(-m[np.intp(P.de)] * y_e)) # Hz
            
            # Compute inhibitory firing rate using sigmoid function (Equation 8)
            # r_i = (a_i * I_i - b_i) / (1 - exp(-d_i * (a_i * I_i - b_i)))
            y_i = m[np.intp(P.ai)] * Ii - m[np.intp(P.bi)]
            ri = y_i / (1.0 - np.exp(-m[np.intp(P.di)] * y_i)) # Hz
            
            # Compute state derivatives 
            # dS_e/dt = -S_e/tau_e + gamma_e * (1 - S_e) * r_e / 1000 (Equation 9)
            dSe = (-Se / m[np.intp(P.tau_e)] + 
                   m[np.intp(P.gamma_e)] * (1.0 - Se) * re)
            
            # dS_i/dt = -S_i/tau_i + gamma_i * r_i (Equation 10)
            dSi = (-Si / m[np.intp(P.tau_i)] + 
                   m[np.intp(P.gamma_i)] * ri)
            
            return np.stack((dSe, dSi)), np.stack((Ie, re))

        return Deco2014_dfun

    # ==========================================================================
    # Steady State Computation Methods
    # ==========================================================================
    
    def _compute_steady_state(self, SC: np.ndarray) -> Tuple[float, float, float, float, float, float, np.ndarray]:
        """
        Compute steady state values for the DMF model.
        
        This method computes the steady state values for all model variables
        either using pre-computed values from Demirtaş et al. (2019) or by
        recomputing them numerically if recompute_steady_state is True.
        
        Args:
            SC: Structural connectivity matrix (N x N)
            
        Returns:
            Tuple containing:
            - Ie_ss: Steady state excitatory current (nA)
            - Ii_ss: Steady state inhibitory current (nA) 
            - re_ss: Steady state excitatory firing rate (Hz)
            - ri_ss: Steady state inhibitory firing rate (Hz)
            - Se_ss: Steady state excitatory synaptic gating
            - Si_ss: Steady state inhibitory synaptic gating
            - J: Computed inhibitory coupling strengths (N,)
        """
        N = len(SC)
        
        # Use pre-computed steady-state values from Demirtaş et al. (2019)
        re_ss = self._SS_RE
        Ie_ss = self._SS_IE
        Se_ss = self._SS_SE
        ri_ss = self._SS_RI
        Ii_ss = self._SS_II
        Si_ss = self._SS_SI

        if self.recompute_steady_state:
            # Recompute steady state for excitatory current
            # From equation (7): r_e = (a_e * I_e - b_e) / (1 - exp(-d_e * (a_e * I_e - b_e)))
            def phi_e_inverse(Ie):
                """Compute r_e - r_e_ss for given I_e."""
                y = self.ae * Ie - self.be
                return y / (1.0 - np.exp(-self.de * y)) - re_ss
            
            try:
                Ie_ss_new = fsolve(phi_e_inverse, x0=Ie_ss)[0]
                if np.isfinite(Ie_ss_new):
                    Ie_ss = Ie_ss_new
            except Exception:
                # Keep default value if optimization fails
                pass

            # Recompute steady state for excitatory synaptic gating
            # From equation (9) at steady state: dS_e/dt = 0
            # Se = gamma_e * tau_e * re / (1000 + gamma_e * tau_e * re)
            Se_ss = (self.gamma_e * self.tau_e * re_ss / self._MS_TO_S / 
                    (1 + self.gamma_e * self.tau_e * re_ss / self._MS_TO_S))

        # Compute inhibitory coupling strength J using steady state values
        # From equation (5): I_e = J_ext_e * I_0 + w * J_NMDA * S_e + J_NMDA * SC@S_e - J * S_i + I_external
        # Solving for J: J = (J_ext_e * I_0 + w * J_NMDA * S_e + J_NMDA * SC@S_e + I_external - I_e) / S_i
        J = ((self.Jext_e * self.I0 +
              self.w * self.J_NMDA * Se_ss +
              self.J_NMDA * SC @ (np.ones(N) * Se_ss) +
              self.I_external - Ie_ss) / Si_ss)

        return Ie_ss, Ii_ss, re_ss, ri_ss, Se_ss, Si_ss, J

    def get_jacobian(self, SC: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian of the DMF model.
        
        This function returns the analytical solution for the Jacobian of the DMF
        with respect to S_e and S_i in the network case.

        Args:
            SC: Structural connectivity matrix (n_roi, n_roi). 
                Note: if using global coupling G, it should be pre-multiplied to SC

        Returns:
            jacobian: The Jacobian matrix (2N x 2N) where N is the number of ROIs

        Notes:
            This method should be executed before calculating the linearized covariance 
            or performing numerical integration, each time the model parameters are modified.
        """
        N = len(SC)

        # Extract biophysical parameters with corrected naming
        tau_e = self.tau_e  
        tau_i = self.tau_i  
        gamma_e = self.gamma_e  
        J_NMDA = self.J_NMDA  
        I0 = self.I0  
        Jext_e = self.Jext_e  
        Jext_i = self.Jext_i  
        w_ee = self.w  
        g_i = self.di  
        b_i = self.bi  
        c_i = self.ai  
        c_e = self.ae  
        b_e = self.be  
        g_e = self.de  

        # Compute steady-state values
        Ie, Ii, re, ri, Se, Si, J = self._compute_steady_state(SC)
        
        # Convert to arrays for vectorized operations
        Ie = Ie * np.ones(N)
        Ii = Ii * np.ones(N)
        re = re * np.ones(N)
        ri = ri * np.ones(N)
        Se = Se * np.ones(N)
        Si = Si * np.ones(N)

        # Initialize Jacobian matrix
        jacobian = np.zeros([N * 2, N * 2])
        
        # Define index arrays for efficient matrix assembly
        first_diag_idx = (np.arange(0, N), np.arange(0, N))  # (Se, Se) block diagonal
        second_diag_idx = (np.arange(0, N), np.arange(N, 2*N))  # (Se, Si) block diagonal
        third_diag_idx = (np.arange(N, 2*N), np.arange(0, N))  # (Si, Se) block diagonal
        fourth_diag_idx = (np.arange(N, 2*N), np.arange(N, 2*N))  # (Si, Si) block diagonal

        # Compute partial derivatives for excitatory population
        # ∂(dSe/dt)/∂Se and related terms
        U = (c_e * Ie - b_e)
        U_star = g_e * U
        # Derivative of firing rate transfer function
        M = ((c_e * (1 - np.exp(-U_star))) - (U * c_e * g_e * np.exp(-U_star))) / ((1 - np.exp(-U_star)) ** 2)
        
        # ∂(dSe/dt)/∂Se - diagonal terms for excitatory-excitatory block
        dSe_dSe = (-1 / tau_e) + gamma_e * (-re + (1 - Se) * M * w_ee * J_NMDA) / self._MS_TO_S
        jacobian[first_diag_idx] = dSe_dSe
        
        # ∂(dSe/dt)/∂Sj - off-diagonal terms for excitatory-excitatory block
        for i in range(N):
            for j in range(N):
                if i != j:
                    dSe_dSj = (1 - Se[i]) * gamma_e * M[i] * J_NMDA * SC[i, j] / self._MS_TO_S
                    jacobian[i, j] = dSe_dSj
        
        # ∂(dSe/dt)/∂Si - diagonal terms for excitatory-inhibitory block
        dSe_dSi = (1 - Se) * gamma_e * M * (-J) / self._MS_TO_S
        jacobian[second_diag_idx] = dSe_dSi

        # Compute partial derivatives for inhibitory population
        # ∂(dSi/dt)/∂Se and ∂(dSi/dt)/∂Si
        U = (c_i * Ii - b_i)
        U_star = g_i * U
        # Derivative of inhibitory firing rate transfer function
        M_i = ((c_i * (1 - np.exp(-U_star))) - (U * g_i * c_i * np.exp(-U_star))) / ((1 - np.exp(-U_star)) ** 2)
        
        # ∂(dSi/dt)/∂Se - diagonal terms for inhibitory-excitatory block
        dSi_dSe = M_i * J_NMDA / self._MS_TO_S
        jacobian[third_diag_idx] = dSi_dSe
        
        # ∂(dSi/dt)/∂Si - diagonal terms for inhibitory-inhibitory block
        dSi_dSi = (-1 / tau_i) - M_i / self._MS_TO_S
        jacobian[fourth_diag_idx] = dSi_dSi

        return jacobian
