# ==========================================================================
# ==========================================================================
# ==========================================================================
# Normal form of a supercritical Hopf bifurcation
#
# General neural mass model known as the normal form of a Hopf bifurcation
# (also known as Landau-Stuart Oscillators), which is the canonical model
# for studying the transition from noisy to oscillatory dynamics.
#
#     .. [Kuznetsov_2013] Kuznetsov, Y.A. "Elements of applied bifurcation theory", Springer Sci & Business
#     Media, 2013, vol. 112.
#
#     .. [Deco_2017]  Deco, G., Kringelbach, M.L., Jirsa, V.K. et al.
#     The dynamics of resting fluctuations in the brain: metastability and its
#     dynamical cortical core. Sci Rep 7, 3095 (2017).
#     https://doi.org/10.1038/s41598-017-03073-5
#
#
# The supHopf model describes the normal form of a supercritical Hopf bifurcation in Cartesian coordinates. This
# normal form has a supercritical bifurcation at $a = 0$ with a the bifurcation parameter in the model. So for
# $a < 0$, the local dynamics has a stable fixed point, and for $a > 0$, the local dynamics enters in a
# stable limit cycle.
#
# The dynamic equations were taken from [Deco_2017]:
#
#         \dot{x}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})x_{i} - omega{i}y_{i} \\
#         \dot{y}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})y_{i} + omega{i}x_{i}
#
#     where a is the local bifurcation parameter and omega the angular frequency.
#
# ==========================================================================
# ==========================================================================
from enum import IntEnum

import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr, AttrType
from neuronumba.numba_tools import addr
from neuronumba.numba_tools.addr import address_as_void_pointer
from neuronumba.numba_tools.types import NDA_f8_1d, NDA_f8_2d
from neuronumba.simulator.models import Model


class Hopf(Model):
    state_vars = Model._build_var_dict(['x', 'y'])
    n_state_vars = len(state_vars)
    c_vars = [0, 1]

    observable_vars = Model._build_var_dict([])
    n_observable_vars = len(observable_vars)

    # ==========================================================================
    # supercritical Hopf bifurcation Constants
    # --------------------------------------------------------------------------
    # Values taken from [Deco_2017]
    a = Attr(default=-0.5, attr_type=AttrType.Model)
    omega = Attr(default=0.3, attr_type=AttrType.Model)
    I_external = Attr(default=0.0, attr_type=AttrType.Model)
    conservative = Attr(default=True, attr_type=AttrType.Model)
    weights = Attr(required=True)
    g = Attr(required=True)

    weights_t = Attr(dependant=True)
    sct = Attr(dependant=True)
    ink = Attr(dependant=True)

    @property
    def get_state_vars(self):
        return Hopf.state_vars

    @property
    def get_observablevars(self):
        return Hopf.observable_vars

    @property
    def get_c_vars(self):
        return Hopf.c_vars

    def _init_dependant(self):
        super()._init_dependant()
        self.weights_t = self.weights.T
        self.ink = self.weights_t.sum(axis=1)

    def initial_state(self, n_rois):
        state = np.empty((Hopf.n_state_vars, n_rois))
        state[0] = 0.1
        state[1] = 0.1
        return state

    def initial_observed(self, n_rois):
        observed = np.empty((1, 1))
        return observed

    # Hopf model has a non-standard coupling
    def get_numba_coupling(self):
        ink = self.ink
        wt = self.weights_t.copy()
        g = self.g

        # TODO: why adding the signature raises a numba warning about state_coupled being a non contiguous array?
        @nb.njit #(nb.f8[:, :](nb.f8[:, :]))
        def hopf_coupling(state: NDA_f8_2d):
            r = np.dot(state, wt)
            return g * (r - ink * state)

        return hopf_coupling

    def get_numba_dfun(self):
        m = self.m.copy()
        P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]))
        def Hopf_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            x = state[0, :]
            y = state[1, :]

            pC = m[np.intp(P.I_external)] + 0j
            # --------------------- From [Deco 2017] original code:
            # First, we need to compute the term (in pseudo-LaTeX notation):
            #       G Sum_i SC_ij (x_i - x_j) =
            #       G (Sum_i SC_ij x_i + Sum_i SC_ij x_j) =
            #       G ((Sum_i SC_ij x_i) + (Sum_i SC_ij) x_j)   <- adding some unnecessary parenthesis.
            # This is implemented in Gus' code as:
            #       wC = we * Cnew;  # <- we is G in the paper, Cnew is SC -> wC = G * SC
            #       sumC = repmat(sum(wC, 2), 1, 2);  # <- for sum Cij * xj == sum(G*SC,2)
            # Thus, we have that:
            #       suma = wC*z - sumC.*z                 # this is sum(Cij*xi) - sum(Cij)*xj, all multiplied by G
            #            = G * SC * z - sum(G*SC,2) * z   # Careful, component 2 in Matlab is component 1 in Python...
            #            = G * (SC*z - sum(SC,2)*z)
            # And now the rest of it...
            # Remember that, in Gus' code,
            #       omega = repmat(2*pi*f_diff',1,2);
            #       omega(:,1) = -omega(:,1);
            # so here I will call omega(1)=-omega, and the other component as + omega
            #       zz = z(:,end:-1:1)  # <- flipped z, because (x.*x + y.*y)     # Thus, this zz vector is (y,x)
            #       dz = a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma               # original formula in the code, using complex numbers z instead of x and y...
            #          = zz * omega   +  z  * (a -  z.* z  - zz.* zz) + suma =    # I will be using vector notation here to simplify ASCII formulae... ;-)
            #          = (y)*(-omega) + (x) * (a - (x)*(x) - (y)*(y)) + suma      # here, (x)*(x) should actually be (x) * (x,y)
            #          =  x *(+omega)    y          y * y     x * x               #        y   y                     (y)
            # ---------------------
            # Calculate the input to nodes due to couplings
            xcoup = coupling[0, :]  # np.dot(SCT,x) - ink * x  # this is sum(Cij*xi) - sum(Cij)*xj
            ycoup = coupling[1, :]  # np.dot(SCT,y) - ink * y  #
            # Integration step
            dx = (m[np.intp(P.a)] - x ** 2 - y ** 2) * x - m[np.intp(P.omega)] * y + xcoup + pC.real
            dy = (m[np.intp(P.a)] - x ** 2 - y ** 2) * y + m[np.intp(P.omega)] * x + ycoup + pC.imag
            return np.stack((dx, dy)), np.empty((1,1))

        return Hopf_dfun
