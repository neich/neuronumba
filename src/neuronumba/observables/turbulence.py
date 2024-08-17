# =======================================================================
# =======================================================================
import numpy as np
from scipy import signal

from neuronumba.basic.attr import Attr
from neuronumba.observables.base_observable import Observable
from neuronumba.tools import matlab_tricks


class Turbulence(Observable):
    """
    Turbulence framework, from:
    Gustavo Deco, Morten L. Kringelbach, Turbulent-like Dynamics in the Human Brain,
    Cell Reports, Volume 33, Issue 10, 2020, 108471, ISSN 2211-1247,
    https://doi.org/10.1016/j.celrep.2020.108471.
    (https://www.sciencedirect.com/science/article/pii/S2211124720314601)

    Part of the Thermodynamics of Mind framework:
    Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
    Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
    https://doi.org/10.1016/j.tics.2024.03.009

    Code by Gustavo Deco, 2020.
    Translated by Marc Gregoris, May 21, 2024
    Refactored by Gustavo Patow, June 9, 2024

    """

    lambda_val = Attr(default=0.18, required=False)
    cog_dist = Attr(required=True)
    c_exp = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self._compute_exp_law()

    def _compute_exp_law(self):
        N = self.cog_dist.shape[0]
        # Compute the distance matrix
        rr = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                rr[i, j] = np.linalg.norm(self.cog_dist[i, :] - self.cog_dist[j, :])
        # Build the exponential-distance matrix
        c_exp = np.exp(-self.lambda_val * rr)
        np.fill_diagonal(c_exp, 1)
        self.c_exp = c_exp

    def _compute_from_fmri(self, bold_signal):
        # bold_signal (ndarray): Bold signal with shape (n_rois, n_time_samples)
        cc = self.compute_turbulence(bold_signal)
        return cc

    def compute_turbulence(self, bold_signal):
        n_rois, t_max = bold_signal.shape
        # Initialization of results-storing data
        enstrophy = np.zeros((n_rois, t_max))
        Phases = np.zeros((n_rois, t_max))

        for seed in range(n_rois):
            Xanalytic = signal.hilbert(bold_signal[seed,:])
            Xanalytic = Xanalytic - np.mean(Xanalytic)
            Phases[seed, :] = np.angle(Xanalytic)

        c_exp = self.c_exp
        for i in range(n_rois):
            sumphases = np.nansum(np.tile(c_exp[i, :], (t_max,1)).T * np.exp(1j * Phases), axis=0) / np.nansum(self.c_exp[i, :])
            enstrophy[i] = np.abs(sumphases)

        Rspatime = np.nanstd(enstrophy)
        Rspa = np.nanstd(enstrophy, axis=1).T
        Rtime = np.nanstd(enstrophy, axis=0)
        acfspa = matlab_tricks.autocorr(Rspa, 100)
        acftime = matlab_tricks.autocorr(Rtime, 100)

        return {
            'Rspatime': Rspatime,
            'Rspa': Rspa.T,
            'Rtime': Rtime,
            'acfspa': acfspa,
            'acftime': acftime,
        }
