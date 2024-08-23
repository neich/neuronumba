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
# import warnings
import numpy as np
# from numba import jit
# import WholeBrain.Observables.BOLDFilters as BOLDFilters
from neuronumba.observables.base_observable import Observable
from neuronumba.basic.attr import Attr
import neuronumba.tools.matlab_tricks as tricks


def InsideOUT(ts, NLAG):
    FowRev = np.zeros((NLAG,))
    AsymFow = np.zeros((NLAG,))
    AsymRev = np.zeros((NLAG,))

    Tm = ts.shape[1]
    for Tau in range(1,NLAG + 1):

        # Compute forward correlation
        ts_1 = ts[:, 0:Tm-Tau]
        ts_2 = ts[:, Tau:Tm]
        FCtau_foward = tricks.corr(ts_1.T, ts_2.T)

        # Compute backwards correlation
        ts_11 = ts[:, Tm-1: Tau - 1: -1].T
        ts_22 = ts[:, Tm - Tau -1:: -1].T
        FCtau_reversal = tricks.corr(ts_11, ts_22)

        # Squeeze to remove unneeded extra dimensions -> not really necessary!
        FCtf = np.squeeze(FCtau_foward)
        FCtr = np.squeeze(FCtau_reversal)

        Itauf = -0.5 * np.log(1 - FCtf**2)
        Itaur = -0.5 * np.log(1 - FCtr**2)
        Reference = (Itauf.flatten() - Itaur.flatten())**2
        threshold = np.quantile(Reference, 0.95)  # Find the indices where the squared difference is greater than the 95th percentile
        index = np.where(Reference > threshold)

        FowRev[Tau-1] = np.nanmean(Reference[index])
        AsymFow[Tau-1] = np.mean(np.abs(Itauf - Itauf.T))
        AsymRev[Tau-1] = np.mean(np.abs(Itaur - Itaur.T))

    return {"FowRev": FowRev, "AsymRev": AsymRev, "AsymFow": AsymFow}


# ================================================================================================================
# Main Insideout class.
# ================================================================================================================
class InsideOut(Observable):

    NLAG = Attr(default=6, required=False)  # Number of taus (lag values) to compute

    def _compute_from_fmri(self, fMRI):
        cc = InsideOUT(fMRI.T, self.NLAG)
        return cc

    # ================================================================================================================
    # calculate_Tauwinner: This method, technically, is not part of the observable, but to keep things coherent, and
    # as it is part of the Framework, we keep it here
    # Computes the group with the max mean FowRev.
    # Dataset is a dict of group labels, and for each label a list of the subjectIDs in FowRev
    # ================================================================================================================
    def calculate_Tauwinner(self, dataset, FowRev):
        max_means = []
        for group in dataset:
            subjects = dataset[group]
            FowRevMatr = np.zeros((self.NLAG, len(subjects)))
            for pos, subject in enumerate(subjects):
                FowRevMatr[:, pos] = FowRev[subject]
            max_means.append(np.argmax(np.mean(FowRevMatr, 1)))
        Tauwinner = np.round(np.mean(max_means))
        return int(Tauwinner)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF