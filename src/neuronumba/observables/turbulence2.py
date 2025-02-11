# =======================================================================
# =======================================================================
import numpy as np

from neuronumba.basic.attr import Attr
from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.observables.turbulence import Turbulence
from neuronumba.tools import matlab_tricks
from scipy.optimize import curve_fit


class Information_transfer(Turbulence):
    """
    Information transfer, extension to the Turbulence framework.

    From:
    Escrichs, A., Perl, Y.S., Uribe, C. et al. Unifying turbulent dynamics framework distinguishes different
    brain states. Commun Biol 5, 638 (2022). https://doi.org/10.1038/s42003-022-03576-6

    Also in:
    Noelia Martínez-Molina, Anira Escrichs, Yonatan Sanz-Perl, Aleksi J. Sihvonen, Teppo Särkämö, Morten L.
    Kringelbach, Gustavo Deco; The evolution of whole-brain turbulent dynamics during recovery from traumatic
    brain injury. Network Neuroscience 2024; 8 (1): 158–177.
    doi: https://doi.org/10.1162/netn_a_00346

    Part of the Thermodynamics of Mind framework:
    Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
    Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
    https://doi.org/10.1016/j.tics.2024.03.009

    Code by Noelia Martínez-Molina, 2024.
    Translated by Gustavo Patow, October 3, 2024
    """
    NR = Attr(default=400, required=False)
    NRini = Attr(default=20, required=False)
    NRfin = Attr(default=80, required=False)

    def _compute_from_fmri(self, bold_signal):
        # bold_signal (ndarray): Bold signal with shape (n_rois, n_time_samples)
        cc = self.compute_information_transfer(bold_signal)
        return cc

    def compute_information_transfer(self, bold_signal):
        n_rois, t_max = bold_signal.shape
        rr_range = np.max(self.rr)
        delta = rr_range / self.NR

        xrange = np.zeros(self.NR)
        for i in range(self.NR):
            xrange[i] = delta / 2 + delta * i

        res = super()._compute_from_fmri(bold_signal)
        entropy = res['enstrophy']
        # Calculate info transfer
        fclam = np.corrcoef(np.squeeze(entropy))

        numind = np.zeros(self.NR)
        fcra = np.zeros(self.NR)
        for i in range(n_rois):
            for j in range(n_rois):
                r = self.rr[i,j]
                index = np.floor(r/delta).astype(int)
                if index == self.NR:
                    index = self.NR - 1
                mcc = fclam[i,j]
                if not np.isnan(mcc):
                    fcra[index] += mcc
                    numind[index] += 1
        # Calculate the slope
        grandcorrfcn = np.divide(fcra, numind)  # element-wise division
        indxx = np.nonzero(~np.isnan(grandcorrfcn))[0]
        indices = indxx[(self.NRini < indxx) & (indxx < self.NRfin)]
        len_indices = len(indices)
        xcoor = np.zeros(len_indices)
        ycoor = np.zeros(len_indices)
        nn = 0
        for k in range(len_indices):
            if grandcorrfcn[k] > 0:
                xcoor[nn] = np.log(xrange[k])
                ycoor[nn] = np.log(grandcorrfcn[k] / grandcorrfcn[indices[0]])
                nn += 1
        xcoor = xcoor[:nn]  # remove unnecessary entries...
        ycoor = ycoor[:nn]
        # least square non-linear curve fitting
        linfunc = lambda x, *A : A[0] * x + A[1]  # fitting function
        A0 = np.array([-1, 1])  # initial parameter
        lb = np.array([-4, -10])  # lower bound
        ub = np.array([4, 10])  # upper bound
        popt, pcov = curve_fit(linfunc, xcoor, ycoor, p0=A0, bounds=(lb, ub), method='trf',  maxfev=10000)
        transfer_sub = abs(popt[0])  # slope
        return res | {
            'Transfer': transfer_sub
        }


class Information_cascade(ObservableFMRI):
    """
    Calculate Information cascade flow and Information cascade.
    Information cascade flow:
       F(\lambda) = <corr_t (R_\lambda(x,t+\Delta t), R_{\lambda-\Delta\lambda}(x,t))>_x
    Information cascade:
       obtained by averaging the information cascade flow across scales λ,
       𝐼 = ⟨F(\lambda)⟩_\lambda

    From
    Rare long-range cortical connections enhance human information processing
    Gustavo Deco, Yonathan Sanz Perl, Peter Vuust1, Enzo Tagliazucchi, Henry Kennedy, Morten L. Kringelbach,
    Volume 31, Issue 20p4436-4448.e5October 25, 2021
    DOI: 10.1016/j.cub.2021.07.064

    Also used in:
    Information cascade and Information cascade flow, extensions to the Turbulence framework, from:
    Escrichs, A., Perl, Y.S., Uribe, C. et al. Unifying turbulent dynamics framework distinguishes different
    brain states. Commun Biol 5, 638 (2022). https://doi.org/10.1038/s42003-022-03576-6

    Noelia Martínez-Molina, Anira Escrichs, Yonatan Sanz-Perl, Aleksi J. Sihvonen, Teppo Särkämö, Morten L.
    Kringelbach, Gustavo Deco; The evolution of whole-brain turbulent dynamics during recovery from traumatic
    brain injury. Network Neuroscience 2024; 8 (1): 158–177.
    doi: https://doi.org/10.1162/netn_a_00346

    Part of the Thermodynamics of Mind framework:
    Kringelbach, M. L., Sanz Perl, Y., & Deco, G. (2024). The Thermodynamics of Mind.
    Trends in Cognitive Sciences (Vol. 28, Issue 6, pp. 568–581). Elsevier BV.
    https://doi.org/10.1016/j.tics.2024.03.009

    Code by Noelia Martínez-Molina, 2024.
    Translated by Gustavo Patow, October 3, 2024
    """

    lambda_values = Attr(default=[0.18], required=False)
    cog_dist = Attr(required=True)

    def _compute_from_fmri(self, bold_signal):
        # bold_signal (ndarray): Bold signal with shape (n_rois, n_time_samples)
        cc = self.compute_information_cascade(bold_signal)
        return cc

    def compute_information_cascade(self, bold_signal):
        turbuRes = {}
        for lambda_v in self.lambda_values:
            # Define and call the turbulence object
            Turbu = Information_transfer(cog_dist=self.cog_dist, lambda_val=lambda_v, ignore_nans=True)
            Turbu.configure()
            turbuRes[lambda_v] = Turbu.from_fmri(bold_signal)
        entropys = {lambda_v: turbuRes[lambda_v]['enstrophy'] for lambda_v in self.lambda_values}
        # Calculate info cascade flow and info cascade
        len_lambdas = len(self.lambda_values)
        TransferLambda = np.zeros(len_lambdas)
        for lambda_pos in range(len_lambdas-1):
            lambda_v = self.lambda_values[lambda_pos]
            lambda_v_next = self.lambda_values[lambda_pos+1]
            cc, pp = matlab_tricks.corr2(np.squeeze(entropys[lambda_v_next][:, 1:]).T,
                                         np.squeeze(entropys[lambda_v][:, :-1]).T)
            TransferLambda[lambda_pos+1] = np.nanmean(np.abs(cc[pp < 0.05]))  # info flow
        InformationCascade = np.nanmean(TransferLambda[1:len_lambdas],axis=0)  # info cascade
        turbus = {f'{attrib}-{lambda_v}': turbuRes[lambda_v][attrib]
                  for attrib in turbuRes[lambda_v] for lambda_v in self.lambda_values}  # This is done to ease serialization...
        return turbus | { # to avoid repeating computations
            'TransferLambda': TransferLambda,  # Information Cascade Flow
            'InformationCascade': InformationCascade  # Information Cascade
        }


