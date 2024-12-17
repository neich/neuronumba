import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import Model, LinearCouplingModel


class Montbrio(LinearCouplingModel):
    state_vars = Model._build_var_dict(["r_e", "r_i", "u_e", "u_i", "S_ee", "S_ie"])
    n_state_vars = len(state_vars)
    c_vars = [0]

    observable_vars = Model._build_var_dict([])
    n_observable_vars = len(observable_vars)

    tau_e = Attr(default=10.0, attributes=Model.Type.Model)
    tau_i = Attr(default=10.0, attributes=Model.Type.Model)
    tau_N = Attr(default=10.0, attributes=Model.Type.Model)
    delta_e = Attr(default=1.0, attributes=Model.Type.Model)
    delta_i = Attr(default=1.0, attributes=Model.Type.Model)
    eta_e = Attr(default=1.0, attributes=Model.Type.Model)
    eta_i = Attr(default=1.0, attributes=Model.Type.Model)
    a_e = Attr(default=0.25, attributes=Model.Type.Model)
    a_i = Attr(default=1.0, attributes=Model.Type.Model)
    g_e = Attr(default=2.5, attributes=Model.Type.Model)
    g_i = Attr(default=0, attributes=Model.Type.Model)
    g_ee = Attr(default=2.5, attributes=Model.Type.Model)
    g_ei = Attr(default=0.0, attributes=Model.Type.Model)
    g_ie = Attr(default=2.5, attributes=Model.Type.Model)
    g_ii = Attr(default=0.0, attributes=Model.Type.Model)

    I_e_ext = Attr(default=0.0, attributes=Model.Type.Model)
    I_i_ext = Attr(default=0.0, attributes=Model.Type.Model)
    J_e = Attr(default=1.0, attributes=Model.Type.Model)
    J_i = Attr(default=0.0, attributes=Model.Type.Model)
    J_A = Attr(default=1.0, attributes=Model.Type.Model)
    J_ee = Attr(default=10.0, attributes=Model.Type.Model)
    J_ei = Attr(default=10.0, attributes=Model.Type.Model)
    J_ie = Attr(default=10.0, attributes=Model.Type.Model)
    J_ii = Attr(default=10.0, attributes=Model.Type.Model)
    J = Attr(default=10.0, attributes=Model.Type.Model)

    J_G_ei = Attr(default=1.0, attributes=Model.Type.Model)
    J_G_ii = Attr(default=1.0, attributes=Model.Type.Model)
    J_N_ee = Attr(default=1.0, attributes=Model.Type.Model)
    J_N_ie = Attr(default=1.0, attributes=Model.Type.Model)

    @property
    def get_state_vars(self):
        return Montbrio.state_vars

    @property
    def get_observablevars(self):
        return Montbrio.observable_vars

    @property
    def get_c_vars(self):
        return Montbrio.c_vars

    def initial_state(self, n_rois):
        state = np.empty((Montbrio.n_state_vars, n_rois))
        state[0] = 0.1
        state[1] = 0.1
        state[2] = 0.1
        state[3] = 0.1
        state[4] = 0.1
        state[5] = 0.1
        return state

    def initial_observed(self, n_rois):
        observed = np.empty((1, 1))
        return observed

    def get_numba_dfun(self):
        m = self.m.copy()
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Montbrio_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            r_e = state[0, :]
            r_i = state[1, :]
            u_e = state[2, :]
            u_i = state[3, :]
            S_ee = state[4, :]
            S_ie = state[5, :]
            c_re = coupling[0, :]

            I_e = m[np.intp(P.I_e_ext)] + (m[np.intp(P.tau_e)] * S_ee) - (m[np.intp(P.J)] * m[np.intp(P.J_G_ei)] * m[np.intp(P.tau_i)] * r_i) + (m[np.intp(P.J_A)] * m[np.intp(P.tau_e)] * c_re)
            I_i = m[np.intp(P.I_i_ext)] + (m[np.intp(P.tau_e)] * S_ie) - (m[np.intp(P.J_G_ii)] * m[np.intp(P.tau_i)] * r_i)

            d_r_e = (m[np.intp(P.delta_e)] / ((np.pi * m[np.intp(P.tau_e)])) + 2.0 * r_e * u_e - m[np.intp(P.g_e)] * r_e) / m[np.intp(P.tau_e)]
            d_r_i = (m[np.intp(P.delta_i)] / ((np.pi * m[np.intp(P.tau_i)])) + 2.0 * r_i * u_i - m[np.intp(P.g_i)] * r_i) / m[np.intp(P.tau_i)]
            d_u_e = (m[np.intp(P.eta_e)] + u_e ** 2 - (r_e * np.pi * m[np.intp(P.tau_e)]) ** 2 + I_e) / m[np.intp(P.tau_e)]
            d_u_i = (m[np.intp(P.eta_i)] + u_i ** 2 - (r_i * np.pi * m[np.intp(P.tau_i)]) ** 2 + I_i) / m[np.intp(P.tau_i)]
            d_S_ee = (-S_ee + m[np.intp(P.J_N_ee)] * r_e) / m[np.intp(P.tau_N)]
            d_S_ie = (-S_ie + m[np.intp(P.J_N_ie)] * r_e) / m[np.intp(P.tau_N)]
            return np.stack((d_r_e, d_r_i, d_u_e, d_u_i, d_S_ee, d_S_ie)), np.empty((1,1))

        return Montbrio_dfun
