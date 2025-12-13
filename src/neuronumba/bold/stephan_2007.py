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
#   Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2007 Aug 15; 42(2): 649–662.
#
# Also, check:
# * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
#   Dynamic causal modelling revisited, NeuroImage 199 (2019) 730–744
# ----------------------------------------
# ----------------------------------------
import numpy as np
from numba import njit

from neuronumba.basic.attr import Attr
from neuronumba.bold.base_bold import Bold


class BoldStephan2007(Bold):
    t_min = Attr(default=20, required=False)  # (s)

    taus = Attr(default=0.65, required=False)  # 0.8;    # time unit (s)  --> kappa in the paper
    tauf = Attr(default=0.41, required=False)  # 0.4;    # time unit (s)  --> gamma in the paper
    tauo = Attr(default=0.98, required=False)  # 1;      # mean transit time (s)  --> tau in the paper
    alpha = Attr(default=0.32, required=False) # 0.2;    # a stiffness exponent   --> alpha in the paper
    itaus = Attr(dependant=True)
    itauf = Attr(dependant=True)
    itauo = Attr(dependant=True)
    ialpha = Attr(dependant=True)

    Eo = Attr(default=0.4, required=False)  # This value is from Obata et al. (2004)
    TE = Attr(default=0.04, required=False)  # --> TE, from Stephan et al. 2007
    vo = Attr(default=0.04, required=False)  # ???
    r0 = Attr(default=25, required=False)  # (s)^-1 --> r0, from Stephan et al. 2007
    theta0 = Attr(default=40.3, required=False)  # (s)^-1
    # Part of equation (12) in Stephan et al. 2007:
    k1 = Attr(dependant=True)
    k2 = Attr(dependant=True)
    k3 = Attr(default=1.0)

    def _init_dependant(self):
        super()._init_dependant()
        self.k1 = 4.3*self.theta0*self.Eo*self.TE
        self.k2 = self.r0*self.Eo*self.TE
        self.itaus = 1.0 / self.taus
        self.itauf = 1.0 / self.tauf
        self.itauo = 1.0 / self.tauo
        self.ialpha = 1.0 / self.alpha


    def compute_bold(self, signal, dt):
        @njit
        def Bold_Stephan2007_compute_bold(signal):
            n_t = signal.shape[0]

            # t_min (sec), dt (ms)
            n_min = int(np.round(t_min * 1000.0/ dt))

            # Convert sampling period to seconds
            dtt = dt / 1000.0
            # Euler method
            # t = t0
            x = np.zeros((4, signal.shape[0], signal.shape[1]))
            # Initial conditions
            x[0, 0, :] = 0
            x[1, 0, :] = 1
            x[2, 0, :] = 1
            x[3, 0, :] = 1
            for n in range(n_t - 1):
                # Equation (9) for s in Stephan et al. 2007
                # Shouldn't it be (0.5 signal[n] + 3) instead of signal[n] ??? also, shouldn't it be taus and tauf instead of itaus and itauf???
                x[0, n + 1] = x[0, n] + dtt * (signal[n] - itaus * x[0, n] - itauf * (x[1, n] - 1))
                # Equation (10) for f in Stephan et al. 2007
                x[1, n + 1] = x[1, n] + dtt * x[0, n]
                # Equation (8) for v and q in Stephan et al. 2007
                x[2, n + 1] = x[2, n] + dtt * itauo * (x[1, n] - x[2, n] ** ialpha)
                x[3, n + 1] = x[3, n] + dtt * itauo * (
                            x[1, n] * (1 - (1 - Eo) ** (1 / x[1, n])) / Eo - (x[2, n] ** ialpha) * x[3, n] / x[2, n])

            # The Balloon-Windkessel model, originally from Buxton et al. 1998:
            # Non-linear BOLD model equations. Page 391. Eq. (13) top in [Stephan2007]
            # t = t[n_min:t.size]
            # s = x[0, n_min:n_t]
            # fi = x[1, n_min:n_t]
            v = x[2, n_min:n_t]
            q = x[3, n_min:n_t]
            b = vo * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))  # Equation (12) in Stephan et al. 2007

            # Code to check whether we have a nan in the arrays...
            # Probably, it comes from a negative value in x[:,1], x[:,2] or x[:,3].
            # If it happens, then directly us the [Stephan2008] implementation:
            # * Klaas Enno Stephan, Lars Kasper, Lee M. Harrison, Jean Daunizeau, Hanneke E.M. den Ouden, Michael Breakspear, and Karl J. Friston
            #   Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2008 Aug 15; 42(2): 649–662.
            #
            # array_sum = np.sum(b)
            # array_has_nan = np.isnan(array_sum)
            # if array_has_nan:
            #     print(f"NAN!!!")

            return b

        t_min = self.t_min

        taus = self.taus
        tauf = self.tauf
        tauo = self.tauo
        alpha = self.alpha
        itaus = self.itaus
        itauf = self.itauf
        itauo = self.itauo
        ialpha = self.ialpha

        Eo = self.Eo
        vo = self.vo
        # Part of equation (12) in Stephan et al. 2007:
        k1 = self.k1
        k3 = self.k3
        k2 = self.k2

        b = Bold_Stephan2007_compute_bold(signal)
        step = int(np.round(self.tr / dt))  # each step is the length of the TR, in milliseconds
        bds = b[step - 1::step, :]
        return bds


class BoldStephan2007Alt(Bold):
    """
    Balloon-Windkessel BOLD model (Stephan et al. 2007).
    Alternative implementation using a standalone njit function with clamping for numerical stability.
    """
    t_min = Attr(default=20, required=False)  # (s)


    taus = Attr(default=0.65, required=False)  # 0.8;    # time unit (s)  --> kappa in the paper
    tauf = Attr(default=0.41, required=False)  # 0.4;    # time unit (s)  --> gamma in the paper
    tauo = Attr(default=0.98, required=False)  # 1;      # mean transit time (s)  --> tau in the paper
    alpha = Attr(default=0.32, required=False) # 0.2;    # a stiffness exponent   --> alpha in the paper


    Eo = Attr(default=0.4, required=False)  # This value is from Obata et al. (2004)
    TE = Attr(default=0.04, required=False)  # --> TE, from Stephan et al. 2007
    vo = Attr(default=0.04, required=False)  # ???
    r0 = Attr(default=25, required=False)  # (s)^-1 --> r0, from Stephan et al. 2007
    theta0 = Attr(default=40.3, required=False)  # (s)^-1


    @override
    def compute_bold(self, signal, dt):
        # dt is in milliseconds, convert to seconds for the function
        b = _stephan_2007_bold_impl(signal, dt / 1000.0, self.t_min,
                                    self.taus, self.tauf, self.tauo, self.alpha,
                                    self.Eo, self.TE, self.vo, self.r0, self.theta0)
        step = int(np.round(self.tr / dt))  # each step is the length of the TR, in milliseconds
        bds = b[step - 1::step, :]
        return bds




@njit
def _stephan_2007_bold_impl(signal, dt, t_min, taus, tauf, tauo, alpha, Eo, TE, vo, r0, theta0):
    """
    Stephan et al. (2007) Balloon-Windkessel model implementation using Numba.
    
    Parameters
    ----------
    signal : ndarray
        Input signal (neural activity) with shape (time_samples, regions).
    dt : float
        Sampling interval in seconds.
    t_min : float
        Initial transient time to discard (s).
    taus : float
        Signal decay time constant (s).
    tauf : float
        Autoregulation time constant (s).
    tauo : float
        Mean transit time (s).
    alpha : float
        Stiffness exponent.
    Eo : float
        Resting oxygen extraction fraction.
    TE : float
        Echo time (s).
    vo : float
        Resting blood volume fraction.
    r0 : float
        Slope of intravascular relaxation rate (Hz).
    theta0 : float
        Frequency offset (Hz).
        
    Returns
    -------
    bold : ndarray
        Simulated BOLD signal with shape (time_samples - n_min, regions).
    """
    n_samples, n_regions = signal.shape
    
    # Compute n_min: number of samples to discard (t_min is in seconds, dt is in seconds)
    n_min = int(np.round(t_min / dt))
    
    # Derived parameters
    k1 = 4.3 * theta0 * Eo * TE
    k2 = r0 * Eo * TE
    k3 = 1.0 
    
    itaus = 1.0 / taus
    itauf = 1.0 / tauf
    itauo = 1.0 / tauo
    ialpha = 1.0 / alpha
    
    # Initial conditions
    s = np.zeros(n_regions)
    f = np.ones(n_regions)
    v = np.ones(n_regions)
    q = np.ones(n_regions)
    
    # Output array (after discarding initial transient)
    n_out = n_samples - n_min
    bold = np.zeros((n_out, n_regions))
    
    for i in range(n_samples):
        # Input signal at this step
        u = signal[i, :]
        
        # Derivatives
        ds = u - itaus * s - itauf * (f - 1.0)
        df = s
        dv = itauo * (f - v**ialpha)
        dq = itauo * (f * (1.0 - (1.0 - Eo)**(1.0 / f)) / Eo - (v**ialpha) * q / v)
        
        # Update state (Euler integration)
        s = s + ds * dt
        f = f + df * dt
        v = v + dv * dt
        q = q + dq * dt


        # Clamp values to avoid numerical instability and NaNs
        # f (flow), v (volume), and q (deoxyhemoglobin) must be positive
        for j in range(n_regions):
            if f[j] < 1e-6:
                f[j] = 1e-6
            if v[j] < 1e-6:
                v[j] = 1e-6
            if q[j] < 1e-6:
                q[j] = 1e-6
        
        # Compute BOLD output after initial transient
        if i >= n_min:
            # Equation (12) in Stephan et al. 2007
            bold[i - n_min, :] = vo * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
        
    return bold




