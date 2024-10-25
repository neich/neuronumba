# -*- coding: utf-8 -*-
# =======================================================================
# Computes the Generative Effective Connectivity
# from
# Morten L. Kringelbach et al. ,Toward naturalistic neuroscience: Mechanisms
# underlying the flattening of brain hierarchy in movie-watching compared to
# rest and task.Sci. Adv.9,eade6049(2023).DOI:10.1126/sciadv.ade6049
#
# Created on Wed Jun 12 16:02:05 2024
# @author: Wiep Francisca Stikvoort, modified by dagush
# Goal: Isolate the gEC fitting in one script
# =======================================================================
# Import necessary packages
import numpy as np
from scipy import signal
from scipy.linalg import expm

from neuronumba.observables.linear.linearfc import LinearFC
from neuronumba.tools import filterps

# time lagged covariance without SC
def calc_COV_emp(tss, timelag=1):
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
    EC = np.zeros((n_roi, n_roi))
    for i in range(n_roi):
        for j in range(n_roi):
            correlation = signal.correlate(tss[i, :] - tss[i, :].mean(), tss[j, :] - tss[j, :].mean(), mode='same')
            lags = signal.correlation_lags(tss[i, :].shape[0], tss[j, :].shape[0], mode='same')
            EC[i, j] = correlation[lags == timelag] / tss.shape[1]
    return EC


def calc_H_freq(all_HC_fMRI, N, Tmax, TR, bpf):
    baseline_ts = np.zeros((len(all_HC_fMRI), N, Tmax))
    for n, subj in enumerate(all_HC_fMRI):
        baseline_ts[n] = all_HC_fMRI[subj]

    # -------------------------- Setup Hopf
    f_diff = filterps.filt_pow_spetra_multiple_subjects(baseline_ts, TR, bpf)
    return 2 * np.pi * f_diff  # omega


def update_EC(eps_fc, eps_cov, FCemp, FCsim, covemp, covsim, SC, only_positive=True):
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
    SCnew *= 0.2

    return SCnew


def calc_sigratio(covsim):
    """
    The calc_sigratio function calculates the normalization factor for the 
    time-lagged covariance matrix. This is used so that the FC, which is a 
    covariance normalized by the standard deviations of the two parts, and the 
    tauCOV are in the same space, dimensionless. 
    
    Parameters
    ----------
    covsim : simulated tss put through calc_EC, format (n_roi,n_roi)

    Returns
    -------
    sigratios in format (n_roi,n_roi)

    """     
    sr = np.zeros((covsim.shape))        
    for i in range(covsim.shape[0]):
        for j in range(covsim.shape[1]):
            sr[i,j] = 1/np.sqrt(abs(covsim[i,i]))/np.sqrt(abs(covsim[j,j]))
    return sr


# --------------- fit gEC
def fitGEC(FC_emp, COV_emp, SC, model, TR):
    # ------ Some constants...
    Tau = 1.0
    G = 1.0
    n_iter = 10000
    olderror = 5000
    epsilon = 1e-5
    its_test = 200
    # ------- number or RoIs
    n_roi = np.shape(SC)[0]

    # To get the simulated FC and EC from the linearized hopf model,
    # to initialise some matrices. Starts with SC and the hopf frequencies
    # hopf_int returns: simulated functional connectivity matrix (FC_sim),
    #                   covariance matrix (COV_sim),
    #                   total covariance matrix (COVsimtotal),
    #                   Jacobian matrix (A)
    A, Qn = model.compute_linear_matrix(SC, 0.01)
    obs = LinearFC()
    result =  obs.from_matrix(A, Qn)
    FC_sim = result['FC']
    COVsimtotal = result['CVth']
    COV_sim = result['CV']

    COV_tausim = np.matmul(expm((Tau * TR) * A), COVsimtotal)  # total simulated covariance at time lag Tau
    COV_tausim = COV_tausim[0:n_roi, 0:n_roi]  # simulated covariance at time lag Tau (nodes of interest)

    # scaling factors based on the simulated and empirical covariance matrices
    sigrat_sim = calc_sigratio(COV_sim)
    sigrat_emp = calc_sigratio(COV_emp)
    newSC = SC

    # In case you want to check the trajectory of the error, intialise some object
    save_err = np.zeros((n_iter))
    save_err_cov = np.zeros((n_iter))
    save_err_FC = np.zeros((n_iter))

    for i in range(n_iter):
        save_err[i] = np.mean((FC_emp - FC_sim) ** 2) + np.mean(((sigrat_emp * COV_emp - sigrat_sim * COV_tausim) ** 2))
        save_err_cov[i] = np.mean(((sigrat_emp * COV_emp - sigrat_sim * COV_tausim) ** 2))
        save_err_FC[i] = np.mean((FC_emp - FC_sim) ** 2)

        # adjust the arguments eps_fc and eps_cov to change the updating of the
        # weights in the gEC depending on the difference between the empirical and
        # simulated FC and time-lagged covariance
        newSC = update_EC(eps_fc=0.000, eps_cov=0.0001, FCemp=FC_emp,
                          FCsim=FC_sim.mean(axis=0), covemp=sigrat_emp * COV_emp,
                          covsim=sigrat_sim * COV_sim, SC=newSC)

        # hopf_int returns: simulated functional connectivity matrix (FC_sim),
        #                   covariance matrix (COV_sim),
        #                   total covariance matrix (COVsimtotal),
        #                   Jacobian matrix (A)
        A, Qn = model.compute_linear_matrix(newSC, 0.01)
        obs = LinearFC()
        result = obs.from_matrix(A, Qn)
        FC_sim = result['FC']
        COVsimtotal = result['CVth']
        COV_sim = result['CV']

        sigrat_sim = calc_sigratio(COV_sim)  # scaling factor based on the simulated covariance matrix
        COV_tausim = np.matmul(expm((Tau * TR) * A), COVsimtotal)  # total simulated covariance at time lag Tau
        COV_tausim = COV_tausim[0:n_roi, 0:n_roi]  # simulated covariance at time lag Tau (nodes of interest)

        if i % its_test < 0.1:
            errornow = save_err[i]
            if (olderror - errornow) / errornow < epsilon:  # if the curent error is smaller than epsilon from last iteration
                save_SC = newSC
                break
            if olderror < errornow:  # if the current error is larger than the one from last iteration
                break
            olderror = errornow  # update old error by current error
        save_SC = newSC
    return save_SC


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF