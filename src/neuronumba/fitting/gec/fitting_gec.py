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
from typing import Union

import numpy as np
from scipy.linalg import expm

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables.lagged_cov import TimeLaggedCOV
from neuronumba.observables.linear.linearfc import LinearFC
from neuronumba.observables import FC
from neuronumba.simulator.compact_bold_simulator import CompactHopfSimulator, CompactDeco2014Simulator

class COV_corr_sim_base(HasAttr):
    tr = Attr(default=3000., doc="TR time in milliseconds")
    n_roi = Attr(default=1, doc="Number of rois")
    tau = Attr(default=1.0)
    # Time-lagged COVariance observable
    cov_obs = Attr(default=TimeLaggedCOV())

    def from_fmri(self, bold_signal):
        # We need two empirical coveriances computed from the timeseries. The non-lag for the computation of the
        # sigratio and the Tau lagged for GEC computation.
        self.cov_obs.tau = self.tau
        COV_emp = np.cov(bold_signal)
        COV_emp_tau = self.cov_obs.from_fmri(bold_signal.T)['t-l-COV']
        # Scaling factors based on the empirical covariance matrices
        sigrat_emp = self.cov_obs.calc_sigratio(COV_emp)
        scaled_COV_emp = sigrat_emp * COV_emp_tau
        return scaled_COV_emp

    def sim(self, SC):
        return self._do_sim(SC)

    def _do_sim(self, SC):
        raise NotImplementedError()


class Linear_COV_corr_sim(COV_corr_sim_base):
    model = Attr(default=None)
    sigma = Attr(default=0.1)

    def _do_sim(self, SC):
        # Compute the model (linear hopf) for this iteration. We get:
        #   FC_sim: simulated functional connectivity matrix
        #   COV_sim: simulatied covaraiance matrix
        #   COVsimtotal: total simulated covariance matrix
        #   A: the Jacobian matrix
        A = self.model.get_jacobian(SC)
        Qn = self.model.get_noise_matrix(self.sigma, len(SC))
        obs = LinearFC()
        result = obs.from_matrix(A, Qn)
        FC_sim = result['FC']
        COVsimtotal = result['CVth']
        COV_sim = result['CV']

        self.cov_obs.tau = self.tau
        sigrat_sim = self.cov_obs.calc_sigratio(COV_sim)  # scaling factor based on the simulated covariance matrix
        COV_sim_tau = np.matmul(expm((self.tau * (self.tr / 1000.0)) * A),
                                COVsimtotal)  # total simulated covariance at time lag Tau
        COV_sim_tau = COV_sim_tau[0:self.n_roi,
                      0:self.n_roi]  # simulated covariance at time lag Tau (nodes of interest)
        scaled_COV_sim = sigrat_sim * COV_sim_tau
        return FC_sim, scaled_COV_sim

class NonLinear_COV_corr_sim(COV_corr_sim_base):
    compact_bold_simulator = Attr(required=True, doc="The compact model simulator to generate the bold signal")
    generated_warmup_samples = Attr(required=True, doc="How many samples required to generate for the warmup (this is the number of signal it will be discarded from the final generated signal)")
    generated_simulated_samples = Attr(required=True, doc="How many useful samples to generate for each bold simulation")
    average_across_simulations_count = Attr(default=1, doc="Stochastic generation is noisy, we can average across multiple generations")

    def _do_sim(self, SC):

        average_count = int(self.average_across_simulations_count)
        if average_count <= 0:
            raise "Invalid `average_across_simulations_count` must be a positive integer"

        # Assign weights and tr to model
        self.compact_bold_simulator.weights = SC
        self.compact_bold_simulator.tr = self.tr

        # Run the model
        FC_sim = None
        COV_sim = None
        for i in range(average_count):
            bold_signal = self.compact_bold_simulator.generate_bold(
                self.generated_warmup_samples,
                self.generated_simulated_samples
            )
            if i == 0:
                FC_sim = FC().from_fmri(bold_signal)['FC']
                COV_sim = self.from_fmri(bold_signal.T)
            else:
                FC_sim += FC().from_fmri(bold_signal)['FC']
                COV_sim += self.from_fmri(bold_signal.T)
        
        # Average runs
        FC_sim = FC_sim / float(average_count)
        COV_sim = COV_sim / float(average_count)

        return FC_sim, COV_sim

class FitGEC(HasAttr):
    class NormMethod:
        MAX = 'MAX'
        STD = 'STD'
        STD_NON_ZERO = 'STD_NON_ZERO'

    # tau = Attr(default=1.0)
    g = Attr(default=1.0)
    max_iters = Attr(default=10000)
    convergence_epsilon = Attr(default=1e-5)
    convergence_test_iters = Attr(default=100)
    # sigma = Attr(default=0.1)
    eps_fc = Attr(default=0.0004)
    eps_cov = Attr(default=0.0001)
    simulator = Attr(default=None)

    norm_method = Attr(default=NormMethod.MAX)
    norm_scaling = Attr(default=0.2)

    # Some debug variable members from last run
    last_run_num_of_iters = 0
    last_run_reason_of_termination = ""
    last_run_convergence_err = None
    last_run_convergence_err_cov = None
    last_run_convergence_err_FC = None
    last_run_compute_time_sec = 0

    def last_run_debug_printing(self):
        """
        Helper function to nicely print debug last computation information on terminal.
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

    def _norm_EC(
        self,
        EC: np.ndarray
    ) -> np.ndarray:
        result = np.copy(EC)

        if self.norm_method == FitGEC.NormMethod.MAX:
            result /= np.max(abs(result))
            result *= self.norm_scaling
        elif self.norm_method == FitGEC.NormMethod.STD:
            result /= np.std(result)
            result *= self.norm_scaling
        elif self.norm_method == FitGEC.NormMethod.STD_NON_ZERO:
            nonzero_mask = result != 0
            if np.sum(nonzero_mask) == 0:
                warning.warn("While normalizing EC all values are zero, returning unchanged")
            else:
                result /= np.std(results[nonzero_mask])
                result *= self.norm_scaling
        else:
            warning.warn("Unkown scaling method, returning unchanged")

        return result

    def _update_EC(
            self,
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
                if SC[i, j] == 0:
                    SCnew[i, j] = 0
        if only_positive == True:
            SCnew[SCnew < 0] = 0
        # SCnew /= np.max(abs(SCnew))
        # SCnew *= maxC
        SCnew = self._norm_EC(SCnew)

        return SCnew

    def fitGEC(
            self,
            timeseries: np.ndarray,
            FC_emp: np.ndarray,
            starting_SC: np.ndarray,
            # model: Model,
            # TR: float
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

        # number or RoIs
        # n_roi = np.shape(starting_SC)[0]

        scaled_COV_emp = self.simulator.from_fmri(timeseries)

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
            # Simulate and compute COV and CORR (FC)
            FC_sim, scaled_COV_sim = self.simulator.sim(newSC)

            # adjust the arguments eps_fc and eps_cov to change the updating of the
            # weights in the gEC depending on the difference between the empirical and
            # simulated FC and time-lagged covariance
            newSC = self._update_EC(eps_fc=self.eps_fc, eps_cov=self.eps_cov, FCemp=FC_emp,
                                      FCsim=FC_sim, covemp=scaled_COV_emp,
                                      covsim=scaled_COV_sim, SC=newSC)

            # Compute errors for this iteration. We saved them for debbuging and such
            self.last_run_convergence_err_FC[i] = np.mean((FC_emp - FC_sim) ** 2)
            self.last_run_convergence_err_cov[i] = np.mean((scaled_COV_emp - scaled_COV_sim) ** 2)
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
                    verbose_stop_reason = f"convergence error increased by {errornow - olderror}"
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
