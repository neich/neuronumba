# =======================================================================
# INSIDEOUT framework, from:
# Deco, G., Sanz Perl, Y., Bocaccio, H. et al. The INSIDEOUT framework
# provides precise signatures of the balance of intrinsic and extrinsic
# dynamics in brain states. Commun Biol 5, 572 (2022).
# https://doi.org/10.1038/s42003-022-03505-7
#
# Part of the Thermodynamics of Mind framework:
# Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
# In Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
# https://doi.org/10.1016/j.tics.2024.03.009
#
# By Gustavo Deco,
# Translated by Marc Gregoris
# Ported by Gustavo Patow
# =======================================================================

import numpy as np
from neuronumba.observables.base_observable import Observable
from neuronumba.basic.attr import Attr
import neuronumba.tools.matlab_tricks as tricks


def inside_out(ts, nlag):
    fow_rev = np.zeros((nlag,))
    asym_fow = np.zeros((nlag,))
    asym_rev = np.zeros((nlag,))

    tm = ts.shape[1]
    for tau in range(1, nlag + 1):

        # Compute forward correlation
        ts_1 = ts[:, 0:tm-tau]
        ts_2 = ts[:, tau:tm]
        fc_tau_forward = tricks.corr(ts_1.T, ts_2.T)

        # Compute backwards correlation
        ts_11 = ts[:, tm-1: tau - 1: -1].T
        ts_22 = ts[:, tm - tau -1:: -1].T
        fc_tau_reversal = tricks.corr(ts_11, ts_22)

        # Squeeze to remove unneeded extra dimensions -> not really necessary!
        fc_tf = np.squeeze(fc_tau_forward)
        fc_tr = np.squeeze(fc_tau_reversal)

        itauf = -0.5 * np.log(1 - fc_tf**2)
        itaur = -0.5 * np.log(1 - fc_tr**2)
        reference = (itauf.flatten() - itaur.flatten())**2
        # Find the indices where the squared difference is greater than the 95th percentile
        threshold = np.quantile(reference, 0.95)
        index = np.where(reference > threshold)

        fow_rev[tau-1] = np.nanmean(reference[index])
        asym_fow[tau-1] = np.mean(np.abs(itauf - itauf.T))
        asym_rev[tau-1] = np.mean(np.abs(itaur - itaur.T))

    return {"FowRev": fow_rev, "AsymRev": asym_rev, "AsymFow": asym_fow}


class InsideOut(Observable):
    """
    Main Insideout class.

    """

    nlag = Attr(default=6, required=False)  # Number of taus (lag values) to compute

    def _compute_from_fmri(self, fmri):
        cc = inside_out(fmri.T, self.nlag)
        return cc

    def calculate_tauwinner(self, dataset, fow_rev):
        """
        This method, technically, is not part of the observable, but to keep things coherent, and
        as it is part of the Framework, we keep it here.
        :param dataset: dict of group labels, and for each label a list of the subjectIDs in fow_rev
        :param fow_rev:
        :return: the group with the max mean fow_rev.
        """
        max_means = []
        for group in dataset:
            subjects = dataset[group]
            fow_rev_matr = np.zeros((self.nlag, len(subjects)))
            for pos, subject in enumerate(subjects):
                fow_rev_matr[:, pos] = fow_rev[subject]
            max_means.append(np.argmax(np.mean(fow_rev_matr, 1)))
        tau_winner = np.round(np.mean(max_means))
        return int(tau_winner)
