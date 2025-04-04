import argparse
import copy
import csv
import gc
import itertools
import os
import time
import sys

import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from numpy.ma.core import repeat

# plt.ion()

from neuronumba.bold import BoldStephan2008
from neuronumba.observables.sw_fcd import SwFCD
from neuronumba.simulator.models import Deco2014
from neuronumba.fitting.fic.fic import FICDeco2014
from neuronumba.simulator.models.montbrio import Montbrio
from neuronumba.tools import filterps, hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import PhFCD, FC

# Local module
from neuronumba.observables.accumulators import ConcatenatingAccumulator, AveragingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic, PearsonSimilarity
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.models.hopf import Hopf
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.tools.loader import load_2d_matrix


def load_subject_list(path):
    subjects = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            subjects.append(int(row[0]))
    return subjects


def save_selected_subjcets(path, subj):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in subj:
            writer.writerow([s])


def load_subjects_data(fmri_path):
    # This function is highly dependent on your input data layout
    # This example is configured for the data downloaded by EBRAINS
    # Read: examples/Data_Raw/ebrains_popovych/README.md
    if not os.path.isdir(fmri_path):
        raise FileNotFoundError(f"Path <{fmri_path}> does not exist or is not a folder!")
    n_sub = 0
    result = {}
    for path in os.listdir(fmri_path):
        subject_path = os.path.join(fmri_path, path)
        if os.path.isdir(subject_path):
            fmri_file = os.path.join(subject_path, 'rfMRI_REST1_LR_BOLD.csv')
            if not os.path.isfile(fmri_file):
                raise FileNotFoundError(f"fMRI file <{fmri_file}> not found!")
            # We want the shape of each fmri to be (n_rois, t_max)
            result[n_sub] = load_2d_matrix(fmri_file).T
            n_sub += 1

    return result

def load_sc(fmri_path, scale):
    if not os.path.isdir(fmri_path):
        raise FileNotFoundError(f"Path <{fmri_path}> does not exist or is not a folder!")
    n_sub = 0
    sc = None
    for path in os.listdir(fmri_path):
        subject_path = os.path.join(fmri_path, path)
        if os.path.isdir(subject_path):
            fmri_file = os.path.join(subject_path, 'Counts.csv')
            if not os.path.isfile(fmri_file):
                raise FileNotFoundError(f"fMRI file <{fmri_file}> not found!")
            m = load_2d_matrix(fmri_file)
            if sc is None:
                sc = m
            else:
                sc = sc + m
            n_sub += 1

    sc = sc / n_sub
    # Warning: we use this normalization for the SC matrix, make sure it makes sense for your own data
    return scale * sc / sc.max()


def simulate(exec_env, g):
    model = exec_env['model']
    weights = exec_env['weights']
    obs_var = exec_env['obs_var']
    t_max_neuronal = exec_env['t_max_neuronal']
    t_warmup = exec_env['t_warmup']
    integrator = exec_env['integrator']
    sampling_period = exec_env['sampling_period']
    model.configure(weights=weights, g=g)
    if 'J' in exec_env:
        model.configure(J=exec_env['J'])

    start_t = time.time()
    signal = simulate_nodelay(model, integrator, weights, obs_var, sampling_period, t_max_neuronal, t_warmup)
    diff_t = time.time() - start_t
    if exec_env['verbose']:
        print(f"Execution time: {diff_t}")
    data_from = int(signal.shape[0] * t_warmup / (t_max_neuronal + t_warmup))
    signal = signal[data_from:, :]
    return signal


def simulate_single_subject(exec_env, g):
    signal = simulate(exec_env, g)
    bold = exec_env['bold']
    sampling_period = exec_env['sampling_period']
    if bold:
        b = exec_env['bold_model']
        bds = b.compute_bold(signal, sampling_period)
    else:
        # Some models like Hopf do not require explicit computation of BOLD signal
        # BUT, we still have to convert the signal into samples of size tr
        tr = exec_env['tr']
        n = int(tr / sampling_period)
        len = signal.shape[0]
        tmp1 = np.pad(signal, ((0, n - len % n), (0, 0)),
                                mode='constant',
                                constant_values=np.nan)
        tmp2 = tmp1.reshape(n, int(tmp1.shape[0]/n), -1)
        bds = np.nanmean(tmp2, axis=0)
    return bds


def process_bold_signals(bold_signals, exec_env):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    num_subjects = len(bold_signals)
    N = bold_signals[next(iter(bold_signals))].shape[1]

    observables = exec_env['observables']

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds in observables:  # Initialize data structs for each observable
        measureValues[ds] = observables[ds][1].init(num_subjects, N)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, bold_signals[s].shape[0], bold_signals[s].shape[1]), end='', flush=True)
        signal = bold_signals[s]  # LR_version_symm(tc[s])
        start_time = time.perf_counter()

        for ds in observables:  # Now, let's compute each measure and store the results
            measure = observables[ds][0]
            accumulator = observables[ds][1]
            if observables[ds][3] is not None:
                bpf = observables[ds][3]
                signal = bpf.filter(signal)
            # FC, swFCD, phFCD, ...
            proc_signal = measure.from_fmri(signal)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, proc_signal[ds])

        print(" -> computed in {} seconds".format(time.perf_counter() - start_time))

    for ds in observables:  # finish computing each observable
        accumulator = observables[ds][1]  # FC, swFCD, phFCD, ...
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def eval_one_param(exec_env, g):
    if 'J_file_name_pattern' in exec_env:
        J_file_name_pattern = exec_env['J_file_name_pattern'].format(np.round(g, decimals=2))
        if os.path.exists(J_file_name_pattern):
            J = hdf.loadmat(J_file_name_pattern)['J']
        else:
            J = FICDeco2014(model=exec_env['model'],
                            obs_var=exec_env['obs_var'],
                            integrator=exec_env['integrator']).compute_J(exec_env['weights'], g)
            hdf.savemat(J_file_name_pattern, {'J': J})
        exec_env['J'] = J
    simulated_bolds = {}
    num_subjects = exec_env['num_subjects']
    for nsub in range(num_subjects):  # trials. Originally it was 20.
        print(f"   Simulating g={g} -> subject {nsub}/{num_subjects}!!!")
        bds = simulate_single_subject(exec_env, g)
        while np.isnan(bds).any() or (np.abs(bds) > np.inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
            raise RuntimeError(f"Numeric error computing subject {nsub}/{num_subjects} for g={g}")
        simulated_bolds[nsub] = bds
        gc.collect()

    dist = process_bold_signals(simulated_bolds, exec_env)
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist['g'] = g

    return dist


# def eval_one_param_multi(exec_env, g):
#     simulated_bolds = {}
#     num_subjects = exec_env['num_subjects']
#     pool = ProcessPoolExecutor(5)
#     results = pool.map(eval_one_param, itertools.repeat((exec_env, g), num_subjects))
#     for i, bds in enumerate(results):
#         if np.isnan(bds).any() or (np.abs(bds) > np.inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
#             raise RuntimeError(f"Numeric error computing subject {i}/{num_subjects} for g={g}")
#         simulated_bolds[i] = bds
#
#     dist = process_bold_signals(simulated_bolds, exec_env)
#     # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
#     dist['g'] = g
#
#     return dist


def compute_g(exec_env, g):
    result = {}
    out_file = exec_env['out_file_name_pattern'].format(np.round(g, decimals=3))
    force_recomputations = False if 'force_recomputations' not in exec_env else exec_env['force_recomputations']
    if not force_recomputations and os.path.exists(out_file):
        print(f"Loading previous data for g={g}")
        sim_measures = hdf.loadmat(out_file)
    else:
        print(f"Starting computation for g={g}")
        sim_measures = eval_one_param(exec_env, g)
        # sim_measures = eval_one_param_multi(exec_env, g)
        hdf.savemat(out_file, sim_measures)

    observables = exec_env['observables']
    processed = exec_env['processed']
    for ds in observables:
        result[ds] = observables[ds][2].distance(sim_measures[ds], processed[ds])
        print(f" {ds} for g={g}: {result[ds]};", end='', flush=True)

    result['g'] = g
    return result


def process_empirical_subjects(bold_signals, observables, bpf=None, verbose=True):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    num_subjects = len(bold_signals)
    # get the first key to retrieve the value of N = number of areas
    n_rois = bold_signals[next(iter(bold_signals))].shape[0]

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds, (_, accumulator, _, _) in observables.items():
        measureValues[ds] = accumulator.init(num_subjects, n_rois)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        # BOLD signals from file have inverse shape
        signal = bold_signals[s].T  # need to be transposed for the rest of NeuroNumba...

        if verbose:
            print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, signal.shape[0],
                                                                    signal.shape[1]), flush=True)

        if bpf is not None:
            signal = bpf.filter(signal)
        for ds, (observable, accumulator, _, _) in observables.items():
            procSignal = observable.from_fmri(signal)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal[ds])

    for ds, (observable, accumulator, _, _) in observables.items():
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nproc", help="Number of parallel processes", type=int, default=10)
    parser.add_argument("--nsubj", help="Number of subject for the simulations", type=int)
    parser.add_argument("--g", help="Single point execution for a global coupling value", type=float)
    parser.add_argument("--g-range", nargs=3, help="Parameter sweep range for G (start, end, step)", type=float)
    parser.add_argument("--bpf", nargs=3, help="Band pass fiter to apply to BOLD signal (k, lp freq, hp freq)", type=float, default=[2, 0.01, 0.1])
    parser.add_argument("--model", help="Model to use (Hopf, Deco2014)", type=str, default='Hopf')
    parser.add_argument("--observable", help="Observable to use (FC, phFCD, swFCD)", type=str, default='phFCD')
    parser.add_argument("--out-path", help="Path to folder for output results", type=str, required=True)
    parser.add_argument("--tr", help="Time resolution of fMRI scanner (seconds)", type=float, required=True)
    parser.add_argument("--sc-scaling", help="Scaling factor for the SC matrix", type=float, default=0.2)
    parser.add_argument("--tmax", help="Override simulation time (seconds)", type=float, required=False)
    parser.add_argument("--fmri-path", help="Path to fMRI timeseries data", type=str, required=True)

    args = parser.parse_args()  # for example, for a single test, use --ge-range 1.0 10.0 1.0

    # This is the data related to the dataset that we are going to load.
    # For your dataset, you need to figure these values

    # Time resolution parameter of the fMRI data (seconds). Each dataset will have its own tr value
    # depending on the scanner setting used to capture the BOLD signal
    tr = args.tr
    # We will discard the first t_min seconds of the simulation
    t_min = 10.0 * tr / 1000.0

    fmris = load_subjects_data(args.fmri_path)
    n_frmis = len(fmris)
    n_rois, t_max = fmris[next(iter(fmris))].shape

    timeseries = np.zeros((n_frmis, n_rois, t_max))
    for i, fmri in enumerate(fmris.values()):
        timeseries[i, :, :] = fmri

    # Compute the simulation length according to input data
    t_max = t_max * tr / 1000.0 if args.tmax is None else args.tmax
    # Compute simulation time in milliseconds
    t_max_neuronal = (t_max + t_min - 1) * 1000.0
    t_warmup = t_min * 1000.0

    # Common integration parameters
    # Integrations step (ms)
    dt = 0.1
    # Sampling period from the raw signal data (ms)
    sampling_period = 1.0

    # Load structural connectivity matrix. In our case, we average all the SC matrices of all subjects
    sc_norm = load_sc(args.fmri_path, args.sc_scaling)

    bold = False
    if args.model == 'Hopf':
        # -------------------------- Setup Hopf frequencies
        bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=tr, apply_detrend=True, apply_demean=True)
        f_diff = filterps.filt_pow_spetra_multiple_subjects(timeseries, tr, bpf)
        omega = 2 * np.pi * f_diff
        model = Hopf(omega=omega, a=-0.02)
        integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2])
        obs_var = 'x'
    elif args.model == 'Deco2014':
        model = Deco2014()
        integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 1e-3])
        bold = True
        obs_var = 're'
    elif args.model == 'Montbrio':
        model = Montbrio()
        integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        bold = True
        obs_var = 'r_e'

    else:
        raise RuntimeError(f"Model <{args.model}> not supported!")

    observable_name = None
    if args.observable == 'FC':
        observable_name = 'FC'
        observables = {observable_name: (FC(), AveragingAccumulator(), PearsonSimilarity(), None)}
    elif args.observable == 'phFCD':
        observable_name = 'phFCD'
        observables = {observable_name: (PhFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic(), None)}
    elif args.observable == 'swFCD':
        observable_name = 'swFCD'
        observables = {observable_name: (SwFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic(), None)}
    else:
        RuntimeError(f"Observable <{args.observable}> not supported!")

    all_fMRI = {s: d for s, d in enumerate(timeseries)}

    out_file_path = os.path.join(args.out_path, f"{args.model}_{observable_name}")
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)

    # Process (or load) empirical data
    emp_filename = os.path.join(out_file_path, 'fNeuro_emp.mat')
    if not os.path.exists(emp_filename):
        bpf_emp = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr, apply_detrend=True, apply_demean=True)
        processed = process_empirical_subjects(all_fMRI, observables, bpf_emp)
        hdf.savemat(emp_filename, processed)
    else:
        processed = {observable_name: load_2d_matrix(emp_filename, index=observable_name)}

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_g{}.mat')

    n_subj = args.nsubj if args.nsubj is not None else n_frmis
    if args.g is not None:
        # Single point execution for debugging purposes
        compute_g({
            'verbose':True,
            'model': copy.deepcopy(model),
            'integrator': copy.deepcopy(integrator),
            'weights': sc_norm,
            'processed': processed,
            'tr': tr,
            'observables': copy.deepcopy(observables),
            'obs_var': obs_var,
            'bold': bold,
            'bold_model': BoldStephan2008().configure(),
            'out_file_name_pattern': out_file_name_pattern,
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
            'force_recomputations': False,
        }, args.g)
    elif args.g_range is not None:
        [g_Start, g_End, g_Step] = args.g_range
        gs = np.arange(g_Start, g_End + g_Step, g_Step)

        # We use parallel processing to compute all the simulations
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
            'bold_model': BoldStephan2008().configure(),
            'out_file_name_pattern': out_file_name_pattern,
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
            'force_recomputations': False,
        } for _ in gs]

        results = []
        while len(gs) > 0:
            pool = ProcessPoolExecutor(max_workers=args.nproc)
            futures = []
            print(f"EXECUTOR --- START cycle for {len(gs)} gs")
            for g in gs:
                exec_env = {
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
                    'bold_model': BoldStephan2008().configure(),
                    'out_file_name_pattern': out_file_name_pattern,
                    'num_subjects': n_subj,
                    't_max_neuronal': t_max_neuronal,
                    't_warmup': t_warmup,
                    'sampling_period': sampling_period
                }
                futures.append((g, pool.submit(compute_g, exec_env, g)))

            while any([not f.done() for _, f in futures]):
                time.sleep(5)

            gs = []
            for g, f in futures:
                try:
                    result = f.result()
                    results.append(result)
                    print(f"EXECUTOR --- FINISHED process for g={g}")
                except Exception as exc:
                    f.cancel()
                    print(f"EXECUTOR --- FAIL. Restarting process for g={g}")
                    gs.append(g)

            pool.shutdown(wait=False,cancel_futures=True)

        fig, ax = plt.subplots()
        rs = sorted(results, key=lambda r: r['g'])
        g = [r['g'] for r in rs]
        data = [r[observable_name] for r in rs]

        ax.plot(g, data)

        ax.set(xlabel=f'G (global coupling) for model {args.model}', ylabel=observable_name,
               title='Global coupling fitting')

        plt.savefig(os.path.join(out_file_path, f'fitting_{args.model}_{observable_name}.png'), dpi=300)

    else:
        RuntimeError("Neither --g not --g-range has been defined")

if __name__ == '__main__':
    # Change False to True if you want to run a full sweep of all combination of models and observables
    if True:
        models = ['Deco2014', 'Hopf', 'Montbrio']
        observables = ['FC', 'phFCD', 'swFCD']
        args = [sys.argv[0], '--nproc', '5', '--g-range', '1', '20', '0.2', '--tr', '720', '--tmax', '600', '--fmri-path', './Data_Raw/ebrains_popovych', '--out-path', './Data_Produced/ebrains_popovych']
        for m, o in list(itertools.product(models, observables)):
            sys.argv = args + ['--model', m, '--observable', o]
            print(f'Running fitting for model {m} and observable {o}')
            run()
    else:
        run()

