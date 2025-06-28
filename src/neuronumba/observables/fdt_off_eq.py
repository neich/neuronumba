# --------------------------------------------------------------------------------------
# Model-free/-based FDT analysis for fMRI signals
#
# Based on the original FDT model-based formulation, presented in:
# [Monti 2025] Fluctuation-dissipation theorem and the discovery of distinctive off-equilibrium signatures of brain states
#    Juan Manuel Monti, Yonatan Sanz Perl, Enzo Tagliazucchi, Morten L. Kringelbach, and Gustavo Deco
#    Phys. Rev. Research 7, 013301 – Published 21 March, 2025
#    DOI: https://doi.org/10.1103/PhysRevResearch.7.013301
#
# This model-free formulation is described in:
# [Patow 2024] Off-Equilibrium Fluctuation-Dissipation Theorem Paves the Way in Alzheimer’s Disease Research
#    Gustavo Patow, Juan Monti, Irene Acero-Pousa, Sebastián Idesis, Anira Escrichs, Yonatan Sanz Perl,
#    Petra Ritter, Morten Kringelbach, Gustavo Deco, the Alzheimer’s Disease Neuroimaging Initiative
#    bioRxiv, doi: https://doi.org/10.1101/2024.09.15.613131
#
# Code by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.signal as signal
from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables.base_observable import Observable, ObservableFMRI
from neuronumba.basic.attr import Attr


# --------------------------------------------------------------------------------------
# FDT_Offeq_Base: class to do the main FDT computations!
# --------------------------------------------------------------------------------------
class FDT_Offeq_Base:
    @staticmethod
    def _analysisFdt2(x, eta, sigma, dt):
        """
        Main equilibrium (FDT) analysis
        This is a more efficient version than the old FDTAnalysis(x, eta, dt)

        :param x: the signal (rois, timepoints)
        :param eta: the noise (rois, timepoints)
        :param sigma: the noise variance
        :param dt: the time step
        :return: the three matrices Cts, Rts, Its
        """
        # Set the noise level
        # print(f'sigma = {sigma}, std(eta)={np.std(eta)}, dt={dt}')
        T = sigma ** 2 / 2.

        # Now, compute variables of interest
        nsim, nsteps = x.shape
        Cts = np.zeros((nsteps, nsteps))
        Rts = np.zeros((nsteps, nsteps))

        for i in range(nsim):
            Cts += np.outer(x[i], x[i])
            Rts += np.outer(x[i], eta[i])

        # Ensemble average
        Cts /= nsim
        Rts /= (nsim * sigma ** 2)

        # Calculates I(t,s) = C(t,t) - C(t,s) - T int_s^t R(t,s)
        Its = np.zeros((nsteps, nsteps))
        for tt in range(nsteps):
            for ss in range(tt):
                tintaux = dt * np.arange(ss, tt + 1)
                Rintaux = Rts[tt, ss:tt + 1]
                intRaux = np.trapz(y=Rintaux, x=tintaux)
                Its[tt, ss] = Cts[tt, tt] - Cts[tt, ss] - T * intRaux  # np.triu(A,1).sum()
        return Cts, Rts, Its

    @staticmethod
    def _computeDistanceFromEquilibrium(I):
        """
        find the distance to the equilibrium state
        :param I: The Integral Violation of the FDT matrix
        :return: the deviation from FDT, namely the integrated I
        """
        absI = np.abs(I)
        mask = np.transpose(np.tri(absI.shape[0], k=0))
        # res = np.zeros(I.shape[0])
        Imask = np.ma.array(absI, mask=mask)
        intI = np.average(Imask)
        return intI


# --------------------------------------------------------------------------------------
# FdtOffeqModelBased
# See [Monti 2025]
# --------------------------------------------------------------------------------------
class Fdt_Offeq_ModelBased(Observable):
    # bold_signal (ndarray): Bold signal with shape (n_time_samples, n_rois)
    bold_signal = Attr(default=None, required=True)
    eta = Attr(required=True) # Noise
    dt = Attr(default=0.1) # Integration time-step

    # Compute entry point
    def _compute(self):
        # ---- No filtering applied, data should be pre-filtered!
        fdt_OffEq = FDT_Offeq_Base()

        # We need the bold signal in (n_rois, n_time_samples)
        x = self.bold_signal.T

        sigma = np.std(self.eta)
        C, R, I = fdt_OffEq._analysisFdt2(x, self.eta/np.sqrt(self.dt), sigma, self.dt)
        intI = fdt_OffEq._computeDistanceFromEquilibrium(I)
        return {'C': C,
                'R': R,
                'I': I,
                'intI': intI}


# --------------------------------------------------------------------------------------
# FdtOffeqModelFree
# See [Patow 2024]
# --------------------------------------------------------------------------------------
class Fdt_Offeq_ModelFree(ObservableFMRI):
    # TR in milliseconds
    tr = Attr(required=True)

    @staticmethod
    def _noiseFilter(fMRI):
        """
        Filter the signal
        :param fMRI: the fMRI signal to process.
        :return: the filtered version of the signal
        """
        N, T = fMRI.shape
        signal_filt = np.zeros_like(fMRI)
        for n in range(N):
            signal_filt[n] = signal.wiener(fMRI[n], 3)  # Apply a Wiener filter to the N-dimensional array fMRI[n].
            # signal_filt[n] = signal.medfilt(fMRI[n], 7)  # Apply a median filter to the input array using a local window-size given by kernel_size. The array will automatically be zero-padded.
            # signal_filt[n] = signal.hilbert(fMRI[n])
        return signal_filt

    @staticmethod
    def _derivative2D(regionSignals, axis=0):
        """
        Derive a function
        :param regionSignals: the signals of the regions ;-)
        :param axis: the axis to apply the derivative to
        :return: the derivative of the region signals
        """
        if axis == 1:
            regionSignals = regionSignals.T
        N, T = regionSignals.shape
        d = np.zeros_like(regionSignals)
        for n in range(N):
            # d[n] = TVRegDiff(regionSignals[n], 10, 100, plotflag=False)
            d[n] = np.gradient(regionSignals[n])
        return d

    @staticmethod
    def _derivative(allSignals, axis=0):
        if allSignals.ndim == 3:
            d = np.zeros_like(allSignals)
            S, N, T = allSignals.shape
            for s in range(S):
                d[s] = Fdt_Offeq_ModelFree._derivative2D(allSignals[s], axis=axis)
        else:
            d = Fdt_Offeq_ModelFree._derivative2D(allSignals, axis=axis)
        return d

    @staticmethod
    def _splitSignal(fMRI):
        """
        Split the signal into main + noise!!!
        :param fMRI:
        :return: The input fMRI split into the signal x,
                 its derivative dxdt,
                 the (non-linear) function Fx,
                 and the noise eta
        """
        # What we want is to decompose the signal so that
        #     dx/dt = -F[x(t)] + eta(t)
        # Let's say that our fMRI signal is x.
        x = fMRI
        # So, let's compute the derivative of our signal, dxdt.
        #     np.gradient uses second order accurate central differences in the interior points and
        #     either first or second order accurate one-sides (forward or backwards) differences at the boundaries.
        dxdt = Fdt_Offeq_ModelFree._derivative(x)
        # Now, we will decompose dxdt into F[x(t)] and eta(t)
        # For this, let's filter our derivative... (here we write mFx = -F[x(t)])
        mFx = Fdt_Offeq_ModelFree._noiseFilter(dxdt)
        Fx = -1 * mFx  # Observe we return Fx, not mFx !!!
        # Finally, the difference signal: it is the subtraction of the filtered signal to the real signal, all WRT the derivative.
        # That is, eta(t) = dx/dt - -F[x(t)]
        eta = dxdt - mFx
        return x, dxdt, Fx, eta

    def _compute(self):
        """
        Model-Free FDT
        :return: A dictionary with the results (The matrices C, R and I, with the integral value of I)
        """
        fdt_OffEq = FDT_Offeq_Base()

        # We need the bold signal in (n_rois, n_time_samples)
        bold_signal = self.bold_signal.T

        # ---- First, let's split the signal into its main part and noise
        x, dxdt, Fx, eta = Fdt_Offeq_ModelFree._splitSignal(bold_signal)
        # ---- Now, do the full FDT pipeline
        dt = self.tr / 1000.
        sigma = np.std(eta)
        C, R, I = fdt_OffEq._analysisFdt2(x, eta, sigma, dt)
        intI = fdt_OffEq._computeDistanceFromEquilibrium(I)
        return {'C': C,
                'R': R,
                'I': I,
                'intI': intI}