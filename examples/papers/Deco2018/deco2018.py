import os

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from global_coupling_fitting import (
    compute_g, process_empirical_subjects, create_observables_dict,
    ModelFactory, BoldModelFactory, IntegratorFactory, plot_fitting_distances
)
from papers.Deco2018.serotonin2A import Deco2018
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools.loader import load_2d_matrix

in_file_path = "./Data_Raw"
out_file_path = "./Data_Produced"

# Register Deco2018 model and its integrator configuration in the factories
ModelFactory.add_model('Deco2018', lambda: Deco2018())
IntegratorFactory.add_integrator_config('Deco2018', lambda dt: EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2]))


def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,90,2)
    even = np.arange(1,90,2)[::-1]  # sort 'descend'
    symLR = np.zeros((90,TC.shape[1]))
    symLR[0:45,:] = TC[odd,:]
    symLR[45:90,:] = TC[even,:]
    return symLR


def sym_LR_EmpiricalSubjects(tc_aal, cond, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        transformed[s] = LR_version_symm(tc_aal[s,cond])
    return transformed


def prepro_G_Optim(sc_norm, all_fMRI):
    # %%%%%%%%%%%%%%% Set General Model Parameters
    wStart = 0.0
    step = 0.1  # 0.025
    wEnd = 2.5 + step
    gs = np.arange(wStart, wEnd, step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)

    dt = 0.1
    sampling_period = 1.0  # Sampling period from the raw signal data (ms)
    n_sim_subj = 15
    t_max_neuronal = 440000
    t_warmup = 0
    tr = 2000.0
    obs_var = 're'

    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=tr)
    observables_list = ['FC,PS', 'swFCD,KS']
    observables = create_observables_dict(observables_list, bpf)

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_g_{}.mat')
    # Use Stephan2007Alt: O(N) memory vs O(4*T*N) in Stephan2007
    # For this config, Stephan2007 allocates a (4, 440000, 90) array = 1.27 GB per subject
    # Stephan2007Alt uses only 4 x (90,) state vectors = ~3 KB
    bold_model = BoldModelFactory.create_model('Stephan2007Alt')

    # Process (or load) empirical data
    emp_filename = os.path.join(out_file_path, 'fNeuro_emp.mat')
    if not os.path.exists(emp_filename):
        processed = process_empirical_subjects(all_fMRI, observables)
        hdf.savemat(emp_filename, processed)
    else:
        processed = {o: load_2d_matrix(emp_filename, index=o) for o in observables.keys()}

    # Single process execution for debugging purposes
    # compute_g({
    #     'verbose': True,
    #     'model': 'Deco2018',
    #     'model_attributes': {'auto_fic': False},
    #     'dt': dt,
    #     'weights': sc_norm,
    #     'processed': processed,
    #     'tr': tr,
    #     'observables': observables_list,
    #     'bpf': bpf,
    #     'obs_var': obs_var,
    #     'bold': True,
    #     'bold_model': bold_model,
    #     'out_file': out_file_name_pattern.format(2.1),
    #     'num_subjects': 1,
    #     't_max_neuronal': t_max_neuronal,
    #     't_warmup': t_warmup,
    #     'sampling_period': sampling_period,
    #     'force_recomputations': True,
    # }, 2.1)  # 2.1 is the reported optimum for this computation.

    # Parallel G sweep using ProcessPoolExecutor with retry on failure.
    # Submit in batches of max_workers to avoid pickling all exec_envs at once:
    # each exec_env contains weights (64KB) + processed (~200KB) that get serialized per task.
    max_workers = 15
    remaining_gs = list(gs)
    finished_gs = []

    while len(remaining_gs) > 0:
        batch = remaining_gs[:max_workers]
        print(f'Creating process pool with {max_workers} workers for batch of {len(batch)}/{len(remaining_gs)} G values')
        pool = ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        future_to_g = {}

        for gf in batch:
            exec_env = {
                'verbose': True,
                'model': 'Deco2018',
                'model_attributes': {'auto_fic': False},
                'dt': dt,
                'weights': sc_norm,
                'processed': processed,
                'tr': tr,
                'observables': observables_list,
                'bpf': bpf,
                'obs_var': obs_var,
                'bold': True,
                'bold_model': bold_model,
                'out_file': out_file_name_pattern.format(np.round(gf, decimals=3)),
                'num_subjects': n_sim_subj,
                't_max_neuronal': t_max_neuronal,
                't_warmup': t_warmup,
                'sampling_period': sampling_period,
                'force_recomputations': True,
            }
            future = pool.submit(compute_g, exec_env, gf)
            future_to_g[future] = gf
            futures.append(future)

        batch_failed = []
        for future in as_completed(futures):
            gf = future_to_g[future]
            try:
                # Don't accumulate results — compute_g already saves to disk,
                # and plot_fitting_distances reads from disk files.
                future.result()
                finished_gs.append(gf)
                print(f"Finished g={gf}")
            except Exception as exc:
                print(f"Failed g={gf}: {exc}")
                batch_failed.append(gf)

        pool.shutdown(wait=True)
        # Re-queue only failed g values from this batch, then continue with remaining
        remaining_gs = batch_failed + remaining_gs[len(batch):]

    plot_fitting_distances(out_file_path, 'fitting_g_*.mat')


def run_model_calibration():
    PLACEBO_cond = 4
    # LSD_cond = 1  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

    sc90 = load_2d_matrix(os.path.join(in_file_path, 'all_SC_FC_TC_76_90_116.mat'), index='sc90')
    C = sc90 / np.max(sc90[:]) * 0.2  # Normalization...

    # mean5HT2A_aalsymm = load_2d_matrix(os.path.join(in_file_path, 'mean5HT2A_bindingaal.mat'), index='mean5HT2A_aalsymm')
    # receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()

    tc_aal = load_2d_matrix(os.path.join(in_file_path, 'LSDnew.mat'), index='tc_aal')
    (N, Tmax) = tc_aal[0, PLACEBO_cond].shape  # N = number of areas; Tmax = total time
    print(f'tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}')

    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset
    tc_transf_PLA = sym_LR_EmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
    # tc_transf_LSD = sym_LR_EmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD

    prepro_G_Optim(C, tc_transf_PLA)


if __name__ == "__main__":
    run_model_calibration()
