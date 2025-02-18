# =======================================================================
# =======================================================================
import numpy as np
from scipy import signal

from neuronumba.basic.attr import Attr
from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.tools import matlab_tricks


class Turbulence(ObservableFMRI):
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
    rr = Attr(dependant=True)

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
        # bold_signal (ndarray): Bold signal with shape (n_rois, n_time_samples)
        cc = self.compute_turbulence(bold_signal)
        return cc

    def compute_turbulence(self, bold_signal):
        n_rois, t_max = bold_signal.shape
        # Initialization of results-storing data
        enstrophy = np.zeros((n_rois, t_max))
        Phases = np.zeros((n_rois, t_max))

        # Hilbert transform to calculate the instantaneous phases per node
        for seed in range(n_rois):
            Xanalytic = signal.hilbert(bold_signal[seed,:])
            Xanalytic = Xanalytic - np.mean(Xanalytic)
            Phases[seed, :] = np.angle(Xanalytic)

        # Calculate Kuramoto LOCAL order parameter for all nodes
        for i in range(n_rois):
            sumphases = np.nansum(np.tile(self.c_exp[i, :], (t_max,1)).T * np.exp(1j * Phases), axis=0) / np.nansum(self.c_exp[i, :])
            enstrophy[i] = np.abs(sumphases)  # Kuramoto local order parameter

        # Calculate Kuramoto global order parameter and metastability for all nodes
        gKoP = np.nanmean(np.abs(np.sum(np.exp(1j * Phases), axis=0)) / n_rois)  # Global Kuramoto parameter (synchronization) for all nodes
        Meta = np.nanstd(np.abs(np.sum(np.exp(1j * Phases), axis=0)) / n_rois)  # Global metastability for all nodes

        R_spa_time = np.nanstd(enstrophy)  # Amplitude turbulence (std of Kuramoto local order parameter across nodes and timepoints)
        R_spa = np.nanstd(enstrophy, axis=1).T  # Amplitude turbulence (std of Kuramoto local order parameter per timepoint across nodes)
        R_time = np.nanstd(enstrophy, axis=0)  # Amplitude turbulence (std of Kuramoto local order parameter per node across timepoints)
        acf_spa = matlab_tricks.autocorr(R_spa, 100)  # Autocorrelation of R in space
        acf_time = matlab_tricks.autocorr(R_time, 100)  # Autocorrelation of R in time

        return {
            'Rspatime': R_spa_time,  # Amplitude turbulence
            'Rspa': R_spa.T,         # Amplitude turbulence across nodes per timepoint
            'Rtime': R_time,         # Amplitude turbulence across timepoints per node
            'acfspa': acf_spa,       # Autocorrelation of R across space
            'acftime': acf_time,     # Autocorrelation of R across time
            'enstrophy': enstrophy,  # Kuramoto local order parameter
            'gKoP': gKoP,            # Global Kuramoto parameter (synchronization)
            'Meta': Meta             # Global metastability
        }
