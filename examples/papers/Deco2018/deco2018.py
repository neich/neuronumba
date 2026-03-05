import argparse
import glob
import os
import re

# import numba
# numba.config.DISABLE_JIT = True  # Disable JIT for debugging purposes; set to False for performance

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from global_coupling_fitting import (
    compute_g, process_empirical_subjects, create_observables_dict,
    ModelFactory, BoldModelFactory, IntegratorFactory, plot_fitting_distances
)
from papers.Deco2018.serotonin2A import Deco2018
from neuronumba.fitting.fic.fic import FICHerzog2022
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.simulator.simulator import simulate_nodelay
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
                'force_recomputations': False,
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


def run_simulation(g):
    sc90 = load_2d_matrix(os.path.join(in_file_path, 'all_SC_FC_TC_76_90_116.mat'), index='sc90')
    C = sc90 / np.max(sc90[:]) * 0.2

    dt = 0.1
    sampling_period = 1.0
    t_max_neuronal = 440000
    t_warmup = 1000

    model = ModelFactory.create_model('Deco2018').set_attributes({'g': g, 'auto_fic': True})
    integrator = IntegratorFactory.create_integrator('Deco2018', dt)

    signal = simulate_nodelay(model, integrator, C, 're',
                              sampling_period=sampling_period,
                              t_max_neuronal=t_max_neuronal,
                              t_warmup=t_warmup)

    print(f'Simulation completed: g={g}, shape={signal.shape} (timepoints x regions)')

    import matplotlib.pyplot as plt
    n_regions = signal.shape[1]
    subsample_step = int(10.0 / sampling_period)  # 1/100 s = 10 ms
    signal_sub = signal[::subsample_step, :]
    time = np.arange(signal_sub.shape[0]) * sampling_period * subsample_step
    fig, ax = plt.subplots(figsize=(14, 8))
    for r in range(n_regions):
        ax.plot(time, signal_sub[:, r], linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing rate (re)')
    ax.set_title(f'Deco2018 simulation — G={g}, {n_regions} regions')
    plt.tight_layout()
    plt.show()


def compute_fic():
    sc90 = load_2d_matrix(os.path.join(in_file_path, 'all_SC_FC_TC_76_90_116.mat'), index='sc90')
    C = sc90 / np.max(sc90[:]) * 0.2

    fitting_files = glob.glob(os.path.join(out_file_path, 'fitting*.mat'))
    if not fitting_files:
        print(f'No fitting*.mat files found in {out_file_path}')
        return

    fic = FICHerzog2022()
    pattern = re.compile(r'fitting_g_([\d.]+)\.mat')

    for f in sorted(fitting_files):
        match = pattern.search(os.path.basename(f))
        if not match:
            print(f'Skipping {f}: cannot extract g value')
            continue
        g = float(match.group(1))
        J = fic.compute_J(C, g)
        out_name = os.path.join(out_file_path, f'J_Balance_we{g}.mat')
        hdf.savemat(out_name, {'J': J})
        print(f'g={g} -> {out_name}  (J shape={J.shape})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deco2018 model workflows')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('calibrate', help='Run G-coupling calibration sweep')

    sim_parser = subparsers.add_parser('simulate', help='Run a single simulation for a given G value')
    sim_parser.add_argument('g', type=float, help='Global coupling parameter value')

    subparsers.add_parser('fic', help='Compute FIC J vectors for all fitted G values')

    args = parser.parse_args()

    if args.command == 'calibrate':
        run_model_calibration()
    elif args.command == 'simulate':
        run_simulation(args.g)
    elif args.command == 'fic':
        compute_fic()
