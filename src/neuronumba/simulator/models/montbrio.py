import numpy as np
import numba as nb
from typing import Dict, List, Tuple

from neuronumba.basic.attr import Attr
from neuronumba.fitting.fic.fic import FICHerzog2022
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model, LinearCouplingModel


class Montbrio(LinearCouplingModel):
    """Montbrio neural mass model implementation.
    
    This class implements the Montbrio neural mass model, which describes the dynamics
    of a population of excitatory and inhibitory neurons. The model includes:
    - Firing rates (r_e, r_i)
    - Mean membrane potentials (u_e, u_i) 
    - Synaptic variables (S_ee, S_ie)
    
    The model is based on the work by Montbrio et al. and includes both local and
    long-range coupling between brain regions.
    """
    
    # State variables
    state_vars = Model._build_var_dict(["r_e", "r_i", "u_e", "u_i", "S_ee", "S_ie"])
    n_state_vars = len(state_vars)
    # Coupling variable is only S_ee
    c_vars = [4]

    observable_vars = Model._build_var_dict([])
    n_observable_vars = len(observable_vars)

    # Automatic FIC computation
    auto_fic = Attr(default=False, attributes=Model.Type.Model,
                   doc="Whether to automatically compute inhibitory coupling strength J using FIC")

    # Time constants (ms)
    tau_e = Attr(default=10.0, attributes=Model.Type.Model)
    tau_i = Attr(default=10.0, attributes=Model.Type.Model)
    tau_N = Attr(default=10.0, attributes=Model.Type.Model)
    
    # Firing rate parameters
    delta_e = Attr(default=1.0, attributes=Model.Type.Model)
    delta_i = Attr(default=1.0, attributes=Model.Type.Model)
    eta_e = Attr(default=1.0, attributes=Model.Type.Model)
    eta_i = Attr(default=1.0, attributes=Model.Type.Model)
    
    # Synaptic parameters
    a_e = Attr(default=0.25, attributes=Model.Type.Model)
    a_i = Attr(default=1.0, attributes=Model.Type.Model)
    g_e = Attr(default=2.5, attributes=Model.Type.Model)
    g_i = Attr(default=0, attributes=Model.Type.Model)
    g_ee = Attr(default=2.5, attributes=Model.Type.Model)
    g_ei = Attr(default=0.0, attributes=Model.Type.Model)
    g_ie = Attr(default=2.5, attributes=Model.Type.Model)
    g_ii = Attr(default=0.0, attributes=Model.Type.Model)

    # External inputs and coupling strengths
    I_e_ext = Attr(default=0.0, attributes=Model.Type.Model)
    I_i_ext = Attr(default=0.0, attributes=Model.Type.Model)
    J_A = Attr(default=1.0, attributes=Model.Type.Model)
    J_ee = Attr(default=10.0, attributes=Model.Type.Model)
    J_ei = Attr(default=10.0, attributes=Model.Type.Model)
    J_ie = Attr(default=10.0, attributes=Model.Type.Model)
    J_ii = Attr(default=10.0, attributes=Model.Type.Model)
    J = Attr(default=10.0, attributes=Model.Type.Model)

    # Dependent parameters
    J_G_ei = Attr(dependant=True, attributes=Model.Type.Model)
    J_G_ii = Attr(dependant=True, attributes=Model.Type.Model)
    J_N_ee = Attr(dependant=True, attributes=Model.Type.Model)
    J_N_ie = Attr(dependant=True, attributes=Model.Type.Model)

    @property
    def get_state_vars(self) -> Dict[str, int]:
        """Get dictionary mapping state variable names to their indices."""
        return Montbrio.state_vars

    @property
    def get_observablevars(self) -> Dict[str, int]:
        """Get dictionary mapping observable variable names to their indices."""
        return Montbrio.observable_vars

    @property
    def get_c_vars(self) -> List[int]:
        """Get list of coupling variable indices."""
        return Montbrio.c_vars

    def _init_dependant(self) -> None:
        """Initialize dependent parameters based on model parameters."""
        super(Montbrio, self)._init_dependant()
        # Calculate synaptic coupling strengths
        self.J_N_ee = self.J_ee + self.g_ee * np.log(self.a_e)
        self.J_N_ie = self.J_ie + self.g_ie * np.log(self.a_e)
        self.J_G_ei = self.J_ei + self.g_ei * np.log(self.a_i)
        self.J_G_ii = self.J_ii + self.g_ii * np.log(self.a_i)
        if self.auto_fic:
            self.J = FICHerzog2022().compute_J(self.weights, self.g)

    def initial_state(self, n_rois: int) -> np.ndarray:
        """Initialize state variables.
        
        Args:
            n_rois: Number of brain regions
            
        Returns:
            Initial state array of shape (n_state_vars, n_rois)
        """
        state = np.empty((Montbrio.n_state_vars, n_rois))
        # Initialize all variables to 0.1
        state.fill(0.0)
        return state

    def initial_observed(self, n_rois: int) -> np.ndarray:
        """Initialize observable variables.
        
        Args:
            n_rois: Number of brain regions
            
        Returns:
            Empty array since no observables in base implementation
        """
        return np.empty((1, 1))

    def get_numba_dfun(self) -> callable:
        """Get the numba-compiled differential function.
        
        Returns:
            Numba-compiled function computing state derivatives
        """
        m = self.m.copy()
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Montbrio_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d) -> Tuple[np.ndarray, np.ndarray]:
            """Compute derivatives of state variables.
            
            Args:
                state: Current state array
                coupling: Coupling input array
                
            Returns:
                Tuple of (state derivatives, empty array for observables)
            """
            # Extract state variables
            r_e = state[0, :]
            r_i = state[1, :]
            u_e = state[2, :]
            u_i = state[3, :]
            S_ee = state[4, :]
            S_ie = state[5, :]
            c_re = coupling[0, :]

            # Compute input currents
            I_e = (m[np.intp(P.I_e_ext)] + 
                  (m[np.intp(P.tau_e)] * S_ee) - 
                  (m[np.intp(P.J)] * m[np.intp(P.J_G_ei)] * m[np.intp(P.tau_i)] * r_i) + 
                  (m[np.intp(P.J_A)] * m[np.intp(P.tau_e)] * c_re))
                  
            I_i = (m[np.intp(P.I_i_ext)] + 
                  (m[np.intp(P.tau_e)] * S_ie) - 
                  (m[np.intp(P.J_G_ii)] * m[np.intp(P.tau_i)] * r_i))

            # Compute derivatives
            d_r_e = ((m[np.intp(P.delta_e)] / (np.pi * m[np.intp(P.tau_e)])) + 
                    2.0 * r_e * u_e - 
                    m[np.intp(P.g_e)] * r_e) / m[np.intp(P.tau_e)]
                    
            d_r_i = ((m[np.intp(P.delta_i)] / (np.pi * m[np.intp(P.tau_i)])) + 
                    2.0 * r_i * u_i - 
                    m[np.intp(P.g_i)] * r_i) / m[np.intp(P.tau_i)]
                    
            d_u_e = ((m[np.intp(P.eta_e)] + 
                     u_e ** 2 - 
                     (r_e * np.pi * m[np.intp(P.tau_e)]) ** 2 + 
                     I_e) / m[np.intp(P.tau_e)])
                     
            d_u_i = ((m[np.intp(P.eta_i)] + 
                     u_i ** 2 - 
                     (r_i * np.pi * m[np.intp(P.tau_i)]) ** 2 + 
                     I_i) / m[np.intp(P.tau_i)])
                     
            d_S_ee = (-S_ee + m[np.intp(P.J_N_ee)] * r_e) / m[np.intp(P.tau_N)]
            d_S_ie = (-S_ie + m[np.intp(P.J_N_ie)] * r_e) / m[np.intp(P.tau_N)]

            return np.stack((d_r_e, d_r_i, d_u_e, d_u_i, d_S_ee, d_S_ie)), np.empty((1, 1))

        return Montbrio_dfun
