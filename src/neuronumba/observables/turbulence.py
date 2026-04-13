# =======================================================================
# turbulence.py
# =======================================================================
import numpy as np
from scipy import signal

from neuronumba.basic.attr import Attr
from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.tools import matlab_tricks


class Turbulence(ObservableFMRI):
    """
    Turbulence framework.

    Reference
    ---------
    Gustavo Deco, Morten L. Kringelbach, Turbulent-like Dynamics in the Human Brain,
    Cell Reports, Volume 33, Issue 10, 2020, 108471, ISSN 2211-1247,
    https://doi.org/10.1016/j.celrep.2020.108471.
    (https://www.sciencedirect.com/science/article/pii/S2211124720314601)

    Part of the Thermodynamics of Mind framework:
    Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
    Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
    https://doi.org/10.1016/j.tics.2024.03.009

    Code by Gustavo Deco, 2020.
    Translated by Marc Gregoris, May 21, 2024.
    Refactored by Gustavo Patow, June 9, 2024.
    """

    lambda_val    = Attr(default=0.18, required=False)
    cog_dist      = Attr(required=True)
    # Dependant outputs populated by _init_dependant
    c_exp         = Attr(dependant=True)
    rr            = Attr(dependant=True)

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
        self.rr = rr
        self.c_exp = c_exp

    def _compute_from_fmri(self, bold_signal):
        # bold_signal (ndarray): Bold signal with shape (n_time_samples, n_rois)
        # transpose for internal use
        return self.compute_turbulence(bold_signal.T)

    def compute_turbulence(self, bold_signal: np.ndarray) -> dict:
        """
        Compute all turbulence-related quantities.

        Parameters
        ----------
        bold_signal : np.ndarray
            Shape (n_rois, n_time_samples).

        Returns
        -------
        dict
            R_spa_time  – amplitude turbulence (scalar)
            R_spa       – turbulence std per node across timepoints, shape (n_rois,)
            R_time      – turbulence std per timepoint across nodes, shape (t_max,)
            acf_spa     – autocorrelation of R_spa
            acf_time    – autocorrelation of R_time
            enstrophy    – Kuramoto local order parameter, shape (n_rois, t_max)
            gKoP        – global Kuramoto parameter (synchronisation)
            Meta        – global metastability
        """
        n_rois, t_max = bold_signal.shape
        enstrophy = np.zeros((n_rois, t_max))
        Phases   = np.zeros((n_rois, t_max))

        # Hilbert transform → instantaneous phase per node
        for seed in range(n_rois):
            Xanalytic = signal.hilbert(bold_signal[seed, :])
            Xanalytic = Xanalytic - np.mean(Xanalytic)
            Phases[seed, :] = np.angle(Xanalytic)

        # Kuramoto LOCAL order parameter
        for i in range(n_rois):
            sumphases = (
                np.nansum(
                    np.tile(self.c_exp[i, :], (t_max, 1)).T
                    * np.exp(1j * Phases),
                    axis=0,
                )
                / np.nansum(self.c_exp[i, :])
            )
            enstrophy[i] = np.abs(sumphases)

        # Global Kuramoto order parameter and metastability
        global_op = np.abs(np.sum(np.exp(1j * Phases), axis=0)) / n_rois
        gKoP = np.nanmean(global_op)
        Meta = np.nanstd(global_op)

        R_spa_time = np.nanstd(enstrophy)
        R_spa      = np.nanstd(enstrophy, axis=1).T
        R_time     = np.nanstd(enstrophy, axis=0)
        acf_spa    = matlab_tricks.autocorr(R_spa, 100)
        acf_time   = matlab_tricks.autocorr(R_time, 100)

        return {
            'R_spa_time': R_spa_time,  # Amplitude turbulence
            'R_spa': R_spa.T,          # Amplitude turbulence across nodes per timepoint
            'R_time': R_time,          # Amplitude turbulence across timepoints per node
            'acf_spa': acf_spa,        # Autocorrelation of R across space
            'acf_time': acf_time,      # Autocorrelation of R across time
            'entrophy': enstrophy,     # Kuramoto local order parameter
            'gKoP': gKoP,              # Global Kuramoto parameter (synchronization)
            'Meta': Meta               # Global metastability
        }
