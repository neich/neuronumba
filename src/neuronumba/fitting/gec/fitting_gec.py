# -*- coding: utf-8 -*-
# =======================================================================
# Computes the Generative Effective Connectivity
# from
# Morten L. Kringelbach et al. ,Toward naturalistic neuroscience: Mechanisms
# underlying the flattening of brain hierarchy in movie-watching compared to
# rest and task.Sci. Adv.9,eade6049(2023).DOI:10.1126/sciadv.ade6049
#
# Created on Wed Jun 12 16:02:05 2024
# @author: Wiep Francisca Stikvoort, modified by dagush & Albert Juncà
# =======================================================================
# Import necessary packages
from email.policy import default
import warnings
import time

import numpy as np
from scipy import signal
from scipy.linalg import expm

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables.linear.linearfc import LinearFC
from neuronumba.tools import filterps
from neuronumba.simulator.models import Model

def calc_H_freq(all_HC_fMRI, N, Tmax, TR, bpf):
    baseline_ts = np.zeros((len(all_HC_fMRI), N, Tmax))
    for n, subj in enumerate(all_HC_fMRI):
        baseline_ts[n] = all_HC_fMRI[subj]

    # -------------------------- Setup Hopf
    f_diff = filterps.filt_pow_spetra_multiple_subjects(baseline_ts, TR, bpf)
    return 2 * np.pi * f_diff  # omega

class FitGEC(HasAttr):
    tau = Attr(default=1.0)
    g = Attr(default=1.0)
    max_iters = Attr(default=10000)
    convergence_epsilon = Attr(default=1e-5)
    convergence_test_iters = Attr(default=100)
    sigma = Attr(default=0.1)
    eps_fc = Attr(default=0.0004)
    eps_cov = Attr(default=0.0001)

    # Some debug variable members from last run
    last_run_num_of_iters = 0
    last_run_reason_of_termination = ""
    last_run_convergence_err = None
    last_run_convergence_err_cov = None
    last_run_convergence_err_FC = None
    last_run_compute_time_sec = 0

    def last_run_debug_printing(self):
        """
        Helper function to nicely print debug on terminal last computation information.
        """
        
        if self.last_run_reason_of_termination == "":
            print("Warning: to debug print first you need to run fitGEC")
            return

        print(
        f"***************************************************************************\n"
        f"FitGEC:\n"
        f"  Number of iterations: {self.last_run_num_of_iters}\n"
        f"  Termination reason: {self.last_run_reason_of_termination}\n"
        f"  Compute time (sec): {self.last_run_compute_time_sec}"
        )

        try:
            import plotext as plt

            print(
            f"  Convergence error plot:\n"
            )

            plt.interactive(False)
            iterations = list(range(len(self.last_run_convergence_err)))
            # plt.clt()  # Clear terminal
            plt.plot(iterations, self.last_run_convergence_err, label="Error")
            plt.plot(iterations, self.last_run_convergence_err_cov, label="Cov Error")
            plt.plot(iterations, self.last_run_convergence_err_FC, label="FC Error")
            plt.title(f"Convergence Error per iter")
            plt.xlabel("Iteration")
            plt.ylabel("Error")
            plt.show()
            # plot_str = plt.build()
            # print(plot_str)
            plt.clear_data()

        except ImportError:
            print("plotext module not installed. Please install it to visualize debug plot convergence errors.")
        
        print(
        f"***************************************************************************"
        )

    # time lagged covariance without SC
    # CAUTION! tss is in (n_roi, time) format
    @staticmethod
    def _calc_COV_emp(tss: np.ndarray, timelag: int = 1):
        """
        wo = without SC mask
        
        Parameters
        ----------
        tss : non-perturbed timeseries, in format (n_roi, n_timesteps)
        timelag : the number of timesteps of your timelag, default = 1
        
        Returns
        -------
        time-lagged cov matrix in format(n_roi, n_roi)
        """
        n_roi = tss.shape[0]
        EC    = np.zeros((n_roi,n_roi))
        for i in range(n_roi):
            for j in range(n_roi):
                correlation = signal.correlate(tss[i,:] - tss[i,:].mean(), tss[j,:] - tss[j,:].mean(), mode = 'full')
                lags        = signal.correlation_lags(tss[i,:].shape[0], tss[j,:].shape[0], mode = 'full')
                EC[i,j]     = correlation[lags == timelag] / tss.shape[1]
        return EC

    @staticmethod
    def _update_EC(
        eps_fc: float, 
        eps_cov: float, 
        FCemp: np.ndarray, 
        FCsim: np.ndarray, 
        covemp: np.ndarray, 
        covsim: np.ndarray, 
        SC: np.ndarray, 
        only_positive: bool = True,
        maxC: float = 0.2):
        """
        Parameters
        ----------
        eps_fc   : parameter, float
        eps_cov  : parameter, float
        FCemp    : empirical functional connectivity, format (n_roi, n_roi)
        FCsim    : simulated functional connectivity, format (n_roi, n_roi)
        covemp   : empirical effective connectivity, format (n_roi, n_roi)
        covsim   : simulated effective connectivity, format (n_roi, n_roi)
        SC       : structural connectivity, format (n_roi, n_roi)
        only_positive : default = True, to keep the update of the SC in positive values
        maxC     : Scaling of the SCnew matrix
                
        Returns
        -------
        An updated SC, format (n_roi, n_roi)

        """
        n_roi = SC.shape[0]
                        
        SCnew = SC + eps_fc * (FCemp - FCsim) + eps_cov * (covemp - covsim)
        for i in range(n_roi):
            for j in range(n_roi):
                if SC[i,j] == 0:
                    SCnew[i,j] = 0
        if only_positive == True:
            SCnew[SCnew < 0] = 0
        SCnew /= np.max(abs(SCnew))
        SCnew *= maxC

        return SCnew

    @staticmethod
    def _calc_sigratio(cov: np.ndarray):
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
                sr[i,j] = 1/np.sqrt(abs(cov[i,i]))/np.sqrt(abs(cov[j,j]))
        return sr

    # --------------- fit gEC
    def fitGEC(
        self, 
        timeseries: np.ndarray, 
        FC_emp: np.ndarray, 
        starting_SC: np.ndarray, 
        model: Model, 
        TR: float
    ):
        """
        Parameters:
            timeseries (matrix 2D): Empirical timeseries data in the format of: (time, regions)
            FC_emp (matrix 2D): Empirical functional connectivity.
            starting_SC (matrix 2D): Starting structural connectivity.
            model: linearized model to use
            TR (float): Repetition time in milliseconds.
        """

        # All the computations inside GEC are performed in the (regions, time) timeseries format. 
        # So convert it before anything else:
        timeseries = timeseries.T

        # Runtime
        start_time = time.time()

        # Perform some checks on the starting_SC.
        # At the moment we are only checking that at least diagonal is zeros, 
        # but more can be added
        if not np.allclose(np.diag(starting_SC), 0):
            warnings.warn("Not all diagonal elemnts in starting_SC are zero.")

        # ------- number or RoIs
        n_roi = np.shape(starting_SC)[0]

        # We need two empirical coveriances computed from the timeseries. The non-lag for the computation of the 
        # sigratio and the Tau lagged for GEC computation. 
        COV_emp = np.cov(timeseries)
        COV_emp_tau = FitGEC._calc_COV_emp(timeseries, timelag=self.tau)
    
        # Scaling factors based on the empirical covariance matrices
        sigrat_emp = FitGEC._calc_sigratio(COV_emp)

        # Initializing SC matrix
        newSC = starting_SC

        # We keep track of the trajectory of the error (e.g. for debbuging)
        self.last_run_convergence_err = np.zeros((self.max_iters))
        self.last_run_convergence_err_cov = np.zeros((self.max_iters))
        self.last_run_convergence_err_FC = np.zeros((self.max_iters))

        # Used to check if we are improving the error on convergency test
        olderror = None

        # Some extra information (mostly for debuging)
        verbose_stop_reason = f"max iterations ({self.max_iters}) reached"

        i = 0
        for i in range(self.max_iters):
            # Compute the model (linear hopf) for this iteration. We get:
            #   FC_sim: simulated functional connectivity matrix
            #   COV_sim: simulatied covaraiance matrix
            #   COVsimtotal: total simulated covariance matrix
            #   A: the Jacobian matrix
            A = model.get_jacobian(newSC)
            Qn = model.get_noise_matrix(self.sigma, len(newSC))
            obs = LinearFC()
            result =  obs.from_matrix(A, Qn)
            FC_sim = result['FC']
            COVsimtotal = result['CVth']
            COV_sim = result['CV']

            sigrat_sim = FitGEC._calc_sigratio(COV_sim)  # scaling factor based on the simulated covariance matrix
            COV_sim_tau = np.matmul(expm((self.tau * (TR / 1000.0)) * A), COVsimtotal)  # total simulated covariance at time lag Tau
            COV_sim_tau = COV_sim_tau[0:n_roi, 0:n_roi]  # simulated covariance at time lag Tau (nodes of interest)

            # adjust the arguments eps_fc and eps_cov to change the updating of the
            # weights in the gEC depending on the difference between the empirical and
            # simulated FC and time-lagged covariance
            newSC = FitGEC._update_EC(  eps_fc=self.eps_fc, eps_cov=self.eps_cov, FCemp=FC_emp,
                                        FCsim=FC_sim, covemp=sigrat_emp * COV_emp_tau,
                                        covsim=sigrat_sim * COV_sim_tau, SC=newSC)

            # Compute errors for this iteration. We saved them for debbuging and such
            self.last_run_convergence_err_FC[i] = np.mean((FC_emp - FC_sim) ** 2)
            self.last_run_convergence_err_cov[i] = np.mean((sigrat_emp * COV_emp_tau - sigrat_sim * COV_sim_tau) ** 2)
            self.last_run_convergence_err[i] = self.last_run_convergence_err_FC[i] + self.last_run_convergence_err_cov[i]        

            # Check for convergence every "convergence_test_iters" times
            if i != 0 and (i % self.convergence_test_iters == 0):
                errornow = self.last_run_convergence_err[i]
                # Stop if no more improvements are made
                if olderror and (((olderror - errornow) / errornow) < self.convergence_epsilon):  
                    save_SC = newSC
                    verbose_stop_reason = f"convergence error reached at epsilon ({self.convergence_epsilon})"
                    break
                # Or if error is growing instate of reducing
                if olderror and olderror < errornow:  
                    verbose_stop_reason = f"convergence error increased by {errornow-olderror}"
                    break
                # update old error by current error
                olderror = errornow  
            save_SC = newSC

        self.last_run_convergence_err = self.last_run_convergence_err[:i]
        self.last_run_convergence_err_cov = self.last_run_convergence_err_cov[:i]
        self.last_run_convergence_err_FC = self.last_run_convergence_err_FC[:i]
        self.last_run_reason_of_termination = f"GEC succesfully computed in {i} iterations. Reason for termination: {verbose_stop_reason}."
        self.last_run_num_of_iters = i
        self.last_run_compute_time_sec = time.time() - start_time
        
        return save_SC


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF