import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessPool

from examples.global_coupling_fitting import compute_g, process_empirical_subjects
from examples.papers.Deco2018.serotonin2A import Deco2018
from neuronumba.bold.stephan_2007 import BoldStephan2007
from neuronumba.observables import FC
from neuronumba.observables.accumulators import ConcatenatingAccumulator, AveragingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic, PearsonSimilarity
from neuronumba.observables.sw_fcd import SwFCD
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools.loader import load_2d_matrix

in_file_path = "./Data_Raw"
out_file_path = "./Data_Produced"

def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,90,2)
    even = np.arange(1,90,2)[::-1]  # sort 'descend'
    symLR = np.zeros((90,TC.shape[1]))
    symLR[0:45,:] = TC[odd,:]
    symLR[45:90,:] = TC[even,:]
    return symLR


def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = LR_version_symm(tc_aal[s,cond])
    return transformed

def prepro_G_Optim(sc_norm, all_fMRI):
    # %%%%%%%%%%%%%%% Set General Model Parameters
    J_fileNames = out_file_path + "J_Balance_we{}.mat"

    distanceSettings = {'FC': (FC, False), 'swFCD': (SwFCD, True)}

    wStart = 0.0
    step = 0.1  # 0.025
    wEnd = 3.0 + step
    gs = np.arange(wStart, wEnd, step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)

    dt = 0.1
    # Sampling period from the raw signal data (ms)
    sampling_period = 1.0
    n_subj = 15
    t_max_neuronal = 440000
    t_warmup = 0
    tr = 2000.0

    model = Deco2018(auto_fic=True)
    integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2])
    bold = True
    obs_var = 're'

    observables = {'FC': (FC(), AveragingAccumulator(), PearsonSimilarity(), None),
                   'swFCD': (SwFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic(), BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=tr))}

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_g{}.mat')

    emp_filename = os.path.join(out_file_path, 'fNeuro_emp.mat')
    if not os.path.exists(emp_filename):
        # bpf_emp = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr, apply_detrend=True, apply_demean=True)
        processed = process_empirical_subjects(all_fMRI, observables)
        hdf.savemat(emp_filename, processed)
    else:
        processed = {o: load_2d_matrix(emp_filename, index=o) for o in observables.keys()}


    # Single process execution for debugging purposes
    # compute_g({
    #     'verbose': True,
    #     'model': copy.deepcopy(model),
    #     'integrator': copy.deepcopy(integrator),
    #     'weights': sc_norm,
    #     'processed': processed,
    #     'tr': tr,
    #     'observables': copy.deepcopy(observables),
    #     'obs_var': obs_var,
    #     'bold': bold,
    #     'bold_model': BoldStephan2007().configure(),
    #     'out_file_name_pattern': out_file_name_pattern,
    #     'num_subjects': 1,
    #     't_max_neuronal': t_max_neuronal,
    #     't_warmup': t_warmup,
    #     'sampling_period': sampling_period
    # }, 3.0)

    pool = ProcessPool(nodes=5)
    rn = list(range(len(gs)))
    # Not entirely sure that the deepcopy() function is needed, but I use it when the object is going to be accessed
    # in read-write mode.
    ee = [{
        'verbose': True,
        'i': i,
        'model': copy.deepcopy(model),
        'integrator': copy.deepcopy(integrator),
        'weights': sc_norm,
        'processed': processed,
        'tr': tr,
        'observables': copy.deepcopy(observables),
        'obs_var': obs_var,
        'bold': bold,
        'bold_model': BoldStephan2007().configure(),
        'out_file_name_pattern': out_file_name_pattern,
        'num_subjects': n_subj,
        't_max_neuronal': t_max_neuronal,
        't_warmup': t_warmup,
        'sampling_period': sampling_period
    } for i, _ in enumerate(rn)]

    results = pool.map(compute_g, ee, gs)
    fig, ax = plt.subplots()
    rs = sorted(results, key=lambda r: r['g'])
    g = [r['g'] for r in rs]
    for o in observables.keys():
        data = [r[o] for r in rs]
        ax.plot(g, data, label=o)
        ax.legend()

    ax.set(xlabel=f'G (global coupling) for model Deco2018',
           title='Global coupling fitting')
    plt.show()


if __name__ == "__main__":
    plt.ion()  # Activate interactive mode

    PLACEBO_cond = 4;
    LSD_cond = 1  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

    sc90 = load_2d_matrix(os.path.join(in_file_path, 'all_SC_FC_TC_76_90_116.mat'), index='sc90')
    C = sc90 / np.max(sc90[:]) * 0.2  # Normalization...

    mean5HT2A_aalsymm = load_2d_matrix(os.path.join(in_file_path, 'mean5HT2A_bindingaal.mat'), index='mean5HT2A_aalsymm')
    receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()

    tc_aal = load_2d_matrix(os.path.join(in_file_path, 'LSDnew.mat'), index='tc_aal')
    (N, Tmax) = tc_aal[1, 1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
    print(f'tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}')

    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset, originally 20...
    print(f"Simulating {NumSubjects} subjects!")

    tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
    # FCemp_cotsampling_PLA = G_optim.processEmpiricalSubjects(tc_transf_PLA, distanceSettings, "Data_Produced/SC90/fNeuro_emp_PLA.mat")
    # FCemp_PLA = FCemp_cotsampling_PLA['FC']; cotsampling_PLA = FCemp_cotsampling_PLA['swFCD'].flatten()

    tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD

    prepro_G_Optim(C, tc_transf_PLA)