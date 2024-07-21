# ----------------------------------------
# ----------------------------------------
# BOLD model parameters. In general, these equations are from:
#
# * K.J. Friston, L. Harrison, and W. Penny,
#   Dynamic causal modelling, NeuroImage 19 (2003) 1273–1302
# * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
#   Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
#
# Later revisited in
# * Klaas Enno Stephan, Lars Kasper, Lee M. Harrison, Jean Daunizeau, Hanneke E.M. den Ouden, Michael Breakspear, and Karl J. Friston
#   Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2008 Aug 15; 42(2): 649–662.
#
# Also, check:
# * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
#   Dynamic causal modelling revisited, NeuroImage 199 (2019) 730–744
# ----------------------------------------
# ----------------------------------------
import numpy as np
# from numba import vectorize, float64
from numba import jit, njit

from neuronumba.basic.attr import Attr, HasAttr


class BoldSthephan2008(HasAttr):
    
    rtol = Attr(
        default=1e-5, 
        required=False)
    atol = Attr(
        default=1e-8, 
        required=False)
    t_min = Attr(
        default=20,  # (s),
        required=False)
    dt = Attr(
        default=0.001,  # (s)
        required=False)

    kappa = Attr(default=0.65, required=False)  # 0.8;    # Rate of vasodilatory signal decay, time unit (s) [Friston2003], eta = Attr(default=0.64, required=False) in [Friston2019]
    gamma = Attr(default=0.41, required=False)  # 0.4;    # Rate of flow-dependent elimination, time unit (s)  [Friston2003], chi = Attr(default=0.32, required=False) in [Friston2019]
    tau = Attr(default=0.98, required=False)  # 1;      # mean transit time (s) in [Friston2003], 1/tau = Attr(default=2, required=False) in [Friston2019]
    alpha = Attr(default=0.32, required=False)  # 0.32; % 0.2;    % Grubb's exponent (a stiffness exponent) [Friston2003] and [Friston2019]
    epsilon = Attr(default=0.34, required=False)  # Intravascular:extravascular ratio (should be one epsilon for each brain area.)
    # [Buxton et al. 1998] used 0.4...
    # Lu and Van Zijl 2005 found values that make it 1 [Stephan et al. 2007]...
    # [Friston2019] initializes this as 1

    Eo = Attr(default=0.4, required=False)  # region-specific resting oxygen extraction fractions. This value is from [Obata et al. 2004]
    TE = Attr(default=0.04, required=False)  # Echo time (seconds), TE, from [Stephan et al. 2007].
    # In [Friston2019] they use phi = Eo*TE as another variable
    vo = Attr(default=0.08, required=False)  # resting venous volume, DCM CODE reads 100*0.08...
    r0 = Attr(default=25, required=False)  # (s)^-1 --> slope r0 of intravascular relaxation rate R_iv as a function of oxygen
    # saturation Y:  R_iv = r0*[(1-Y)-(1-Y0)]. This value of r0 from [Stephan et al. 2007]
    theta0 = Attr(default=40.3, required=False)  # (s)^-1, frequency offset at the outer surface of magnetized vessels

    def compute_required_vector_lenght(self, T):
        n_min = int(np.round(self.t_min / self.dt))
        n_t = int(T/self.dt)
        return n_t - n_min

    def compute_bold(self, T, x):
        @njit()
        def isclose(a, b):
            return np.absolute(a - b) <= (atol + rtol * np.absolute(b))

        @njit
        def BOLDModel(T, x):
            # The Hemodynamic model with one simplified neural activity
            #
            # T          : total time (s)
            # x          : the input neural activity
            #
            # Based on the paper:
            # Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
            # Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
            #
            # With the log-transformation of the equations described in
            # Klaas Enno Stephan, Lars Kasper, Lee M. Harrison, Jean Daunizeau, Hanneke E.M. den Ouden, Michael Breakspear, and Karl J. Friston
            # Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2008 Aug 15; 42(2): 649–662.
            #
            # In this paper [Stephan2008], for any of the 4 equations z={s,f,v,q} in the form dz/dt = F(z),
            # they introduce the change of variables ztilde = ln(z)  <=>  z = exp(ztilde).
            # Thus, the equations are now dztilde/dt = F(z)/z, and we solve for ztilde and afterwards we recover z from ztilde,
            # but this time ensuring a proper support for these non-negative states and numerical stability when evaluating
            # these equations during parameter estimation.
            # Warning! The DCM CODE does not apply the change to s, only to {f,v,q} !!!

            # global t_min, dt
            # global Eo
            itau = 1 / tau
            ialpha = 1 / alpha

            N = x.shape[1]
            # Euler method
            s = np.zeros((n_t, N));
            f = np.zeros((n_t, N));
            ftilde = np.zeros((n_t, N))
            v = np.zeros((n_t, N));
            vtilde = np.zeros((n_t, N))
            q = np.zeros((n_t, N));
            qtilde = np.zeros((n_t, N))
            # Initial conditions x0 = np.array([0, 1, 1, 1]) <- they should have been all 0...
            s[0] = 1
            f[0] = 1
            v[0] = 1
            q[0] = 1
            ftilde[0] = 0
            vtilde[0] = 0
            qtilde[0] = 0  # stilde[0] = 0
            for n in range(n_t - 1):
                # Equation (9) for s in [Stephan et al. 2007]. This should be changed to eq. A6 in [Stephan2008]
                # Warning! [Friston2019] and the DCM Code multiplies x[n] by the input efficacies,
                # as in the CODE: P(7:end)'*u(:), which is the same as Sum_i \beta_i x_i
                # Equation A6 in [Stephan2008] reads:
                # stilde[n+1] = stilde[n] + dt * ((x[n] - kappa * s[n] - gamma * (f[n] - 1))/s[n])
                # Warning! The DCM code applies this change of variables only to {f,v,q}, not s...
                # This means that equation A6 in [Stephan2008] is NOT used, but Equation (9) in [Stephan2007] instead...
                # Changes in vasodilatory signalling s:
                s[n + 1] = s[n] + dt * (x[n] - kappa * s[n] - gamma * (f[n] - 1))
                # Equation (10) for f in [Stephan et al. 2007]. Now, changed to eq. A7 in [Stephan2008]
                # Changes in blood flow f :
                np.clip(f[n], 1, None)
                ftilde[n + 1] = ftilde[n] + dt * (s[n] / f[n])
                # Equation (8)-1st for v in [Stephan et al. 2007]. Now, changed to eq. A8 in [Stephan2008]
                # Changes in venous blood volume v:
                # if isclose(v[n], 0.):
                #     # print("v[n] is close to 0")
                #     v[n] = 1e-8
                fv = v[n] ** ialpha  # outflow
                vtilde[n + 1] = vtilde[n] + dt * ((f[n] - fv) / (tau * v[n]))
                # Equation (8)-2nd for q in [Stephan et al. 2007]. Now, changed to eq. A9 in [Stephan2008]
                # Changes in deoxyhemoglobin content q:
                np.clip(q[n], 0.01, None)
                ff = (1 - (1 - Eo) ** (1 / f[n])) / Eo  # oxygen extraction
                qtilde[n + 1] = qtilde[n] + dt * ((f[n] * ff - fv * q[n] / v[n]) / (tau * q[n]))

                # Now, exponentiate to get the "good" hemodynamic variables...
                # s[n+1] = np.exp(stilde[n+1])
                f[n + 1] = np.exp(ftilde[n + 1])
                v[n + 1] = np.exp(vtilde[n + 1])
                q[n + 1] = np.exp(qtilde[n + 1])
                # ============== DEBUG CODE, enable only in non-Numba mode!
                # if np.isnan([f,v,q]).any():
                #     print(f'full NaN stop @ {n}!')
                # if np.isinf([f,v,q]).any():
                #     print(f'full inf stop @ {n}!')

            # Equation (12) in [Stephan et al. 2007]: simulated BOLD response to input
            k1 = 4.3 * theta0 * Eo * TE  # [Friston2019] uses 6.9 phi, which matches exactly 4.3*theta0/r0 = 6.9
            k2 = epsilon * r0 * Eo * TE  # [Friston2019] uses epsilon * phi, obviously phi = Eo*TE
            k3 = 1 - epsilon  # [Friston2019] uses 1-epsilon
            # The Balloon-Windkessel model, originally from [Buxton et al. 1998]:
            vv = v[n_min:n_t]
            qq = q[n_min:n_t]
            # if isclose(vv, 0.).any():
            #    print("vv is close to 0 !!!")
            vv[isclose(vv, 0.)] = 1e-8
            b = vo * (k1 * (1 - qq) + k2 * (1 - qq / vv) + k3 * (1 - vv))  # Equation (12) in Stephan et al. 2007

            return b


        n_t = int(T / self.dt)
        tau = self.tau
        n_min = int(np.round(self.t_min / self.dt))
        alpha = self.alpha
        dt = self.dt
        kappa = self.kappa
        gamma = self.gamma
        Eo = self.Eo
        TE = self.TE
        vo = self.vo
        r0 = self.r0
        theta0 = self.theta0
        epsilon = self.epsilon
        atol = self.atol
        rtol = self.rtol

        return BOLDModel(T, x)







