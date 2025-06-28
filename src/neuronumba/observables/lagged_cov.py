import numpy as np
from scipy import signal

from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.basic.attr import Attr

class TimeLaggedCOV(ObservableFMRI):
    """
    Time-lagged COV class.

    Args:
        fmri (ndarray): Bold signal with shape (n_time_samples, n_rois)
        tau (
    """
    tau = Attr(default=1)  # lag in timepoints (tr)

    def _compute_from_fmri(self, fmri):
        ec = TimeLaggedCOV._calc_COV_emp(fmri.T, self.tau)
        return {'t-l-COV': ec}

    # time lagged covariance without SC
    # CAUTION! tss is in (n_roi, time) format
    @staticmethod
    def _calc_COV_emp(tss: np.ndarray, timelag: int = 1):
        """
        Parameters
        ----------
        tss : non-perturbed timeseries, in format (n_roi, n_timesteps)
        timelag : the number of timesteps of your timelag, default = 1

        Returns
        -------
        time-lagged cov matrix in format(n_roi, n_roi)
        """
        n_roi = tss.shape[0]
        EC = np.zeros((n_roi, n_roi))
        for i in range(n_roi):
            for j in range(n_roi):
                correlation = signal.correlate(tss[i, :] - tss[i, :].mean(), tss[j, :] - tss[j, :].mean(), mode='full')
                lags = signal.correlation_lags(tss[i, :].shape[0], tss[j, :].shape[0], mode='full')
                EC[i, j] = correlation[lags == timelag] / tss.shape[1]
        return EC

    @staticmethod
    def calc_sigratio(cov: np.ndarray):
        """
        The calc_sigratio function calculates the normalization factor for the
        time-lagged covariance matrix. This is used so that the FC, which is a
        covariance normalized by the standard deviations of the two parts, and the
        tauCOV are in the same space, dimensionless.

        Parameters
        ----------
        cov : tss put through calc_EC, format (n_roi,n_roi)

        Returns
        -------
        sigratios in format (n_roi,n_roi)

        """
        sr = np.zeros((cov.shape))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                sr[i, j] = 1 / np.sqrt(abs(cov[i, i])) / np.sqrt(abs(cov[j, j]))
        return sr

