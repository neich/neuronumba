import argparse
import copy
import csv
import os
import time

import h5py
import numpy as np
from pathos.multiprocessing import ProcessPool

from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.models import Deco2014
from neuronumba.tools import filterps, hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import PhFCD, FC

# Local module
from neuronumba.observables.accumulators import ConcatenatingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.models.hopf import Hopf
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.tools.loader import load_2d_matrix


def read_matlab_h5py(filename):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]
        # get the object type for a_group_key: usually group or dataset
        # print(type(f['subjects_idxs']))
        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array

        all_fMRI = {}
        subjects = list(f['subject'])
        for pos, subj in enumerate(subjects):
            print(f'reading subject {pos}')
            group = f[subj[0]]
            dbs80ts = np.array(group['dbs80ts'])
            all_fMRI[pos] = dbs80ts.T

    return all_fMRI

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
            fmri_file = os.path.join(subject_path, 'rFMRI_REST1_LR_BOLD.csv')
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
        b = BoldStephan2008()
        bds = b.compute_bold(signal, sampling_period)
    else:
        # Some models like Hopf do not require explicit computation of BOLD signal
        # BUT, we still have to convert the signal into samples of size tr
        tr = exec_env['tr']
        n = int(tr * 1000.0 / sampling_period)
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
    tr = exec_env['tr']

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
            bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=tr)
            bold_filt = bpf.filter(signal)
            # FC, swFCD, phFCD, ...
            proc_signal = measure.from_fmri(bold_filt)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, proc_signal[ds])

        print(" -> computed in {} seconds".format(time.perf_counter() - start_time))

    for ds in observables:  # finish computing each observable
        accumulator = observables[ds][1]  # FC, swFCD, phFCD, ...
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def eval_one_param(exec_env, g):
    simulated_bolds = {}
    num_subjects = exec_env['num_subjects']
    for nsub in range(num_subjects):  # trials. Originally it was 20.
        print(f"   Simulating g={g} -> subject {nsub}/{num_subjects}!!!")
        bds = simulate_single_subject(exec_env, g)
        while np.isnan(bds).any() or (np.abs(bds) > np.inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
            raise RuntimeError(f"Numeric error computing subject {nsub}/{num_subjects} for g={g}")
        simulated_bolds[nsub] = bds

    observables = exec_env['observables']
    dist = process_bold_signals(simulated_bolds, exec_env)
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist['g'] = g

    return dist


def compute_g(exec_env, g):
    result = {}
    out_file = exec_env['out_file_name_pattern'].format(np.round(g, decimals=3))
    if os.path.exists(out_file):
        print(f"Loading previous data for g={g}")
        sim_measures = hdf.loadmat(out_file)
    else:
        print(f"Starting computation for g={g}")
        sim_measures = eval_one_param(exec_env, g)
        hdf.savemat(out_file, sim_measures)

    observables = exec_env['observables']
    processed = exec_env['processed']
    for ds in observables:
        result[ds] = observables[ds][2].distance(sim_measures[ds], processed[ds])
        print(f" {ds} for g={g}: {result[ds]};", end='', flush=True)

    return result


def process_empirical_subjects(bold_signals, observables, bpf, verbose=True):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    num_subjects = len(bold_signals)
    # get the first key to retrieve the value of N = number of areas
    n_rois = bold_signals[next(iter(bold_signals))].shape[0]

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds, (_, accumulator, _) in observables.items():
        measureValues[ds] = accumulator.init(num_subjects, n_rois)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        if verbose:
            print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, bold_signals[s].shape[0],
                                                                    bold_signals[s].shape[1]), flush=True)
        # BOLD signals from file have inverse shape
        signal = bold_signals[s].T  # LR_version_symm(tc[s])

        signal_filt = bpf.filter(signal)
        for ds, (observable, accumulator, _) in observables.items():
            procSignal = observable.from_fmri(signal_filt)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal[ds])

    for ds, (observable, accumulator, _) in observables.items():
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--g-range", nargs=3, help="Parameter sweep range for G (start, end, step)", type=float, required=True)
    parser.add_argument("--model", help="Model to use (Hopf, Deco2014)", type=str, default='Hopf')
    parser.add_argument("--out-path", help="Path to folder for output results", type=str, required=True)
    parser.add_argument("--tr", help="Time resolution of fMRI scanner (seconds)", type=float, required=True)
    parser.add_argument("--sc-scaling", help="Scaling factor for the SC matrix", type=float, default=0.2)
    parser.add_argument("--tmax", help="Override simulation time (seconds)", type=float, required=False)
    parser.add_argument("--fmri-path", help="Path to fMRI timeseries data", type=str, required=True)

    args = parser.parse_args()  # for example, for a single test, use --ge-range 1.0 10.0 1.0

    [g_Start, g_End, g_Step] = args.g_range

    # This is the data related to the dataset that we are going to load.
    # For your dataset, you need to figure these values

    # Time resolution parameter of the fMRI data (seconds). Each dataset will have its own tr value
    # depending on the scanner setting used to capture the BOLD signal
    tr = args.tr
    # We will discard the first t_min seconds of the simulation
    t_min = 10.0 * tr

    fmris = load_subjects_data(args.fmri_path)
    n_subj = len(fmris)
    n_rois, t_max = fmris[next(iter(fmris))].shape

    timeseries = np.zeros((n_subj, n_rois, t_max))
    for i, fmri in enumerate(fmris.values()):
        timeseries[i, :, :] = fmri

    # Compute the simulation length according to input data
    t_max = t_max * tr if args.tmax is None else args.tmax
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
        integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2])
        bold = True
        obs_var = 're'
    else:
        raise RuntimeError(f"Model <{args.model}> not supported!")

    # observable_name = 'FC'
    observable_name = 'phFCD'

    # Observable definition
    # observables = {observable_name: (FC(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic())}
    observables = {observable_name: (PhFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic())}

    all_fMRI = {s: d for s, d in enumerate(timeseries)}

    # Process (or load) empirical data
    emp_filename = os.path.join(args.out_path, 'fNeuro_emp.mat')
    if not os.path.exists(emp_filename):
        bpf_emp = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr, apply_detrend=True, apply_demean=True)
        processed = process_empirical_subjects(all_fMRI, observables, bpf_emp)
        hdf.savemat(emp_filename, processed)
    else:
        processed = {observable_name: load_2d_matrix(emp_filename, index=observable_name)}

    # -------------------------- Preprocessing pipeline!!!
    gs = np.arange(g_Start, g_End + g_Step, g_Step)
    out_file_name_pattern = os.path.join(args.out_path, 'fitting_g{}.mat')


    # Single point execution for debugging purposes
    # compute_g({
    #     'verbose':True,
    #     'model': copy.deepcopy(model),
    #     'integrator': copy.deepcopy(integrator),
    #     'weights': sc_norm,
    #     'processed': processed,
    #     'tr': tr,
    #     'observables': copy.deepcopy(observables),
    #     'obs_var': obs_var,
    #     'bold': bold,
    #     'out_file_name_pattern': out_file_name_pattern,
    #     'num_subjects': n_subj,
    #     't_max_neuronal': t_max_neuronal,
    #     't_warmup': t_warmup,
    #     'sampling_period': sampling_period
    # }, 2.0)
    # exit(0)

    # We use parallel processing to compute all the simulations
    pool = ProcessPool(nodes=10)
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
        'out_file_name_pattern': out_file_name_pattern,
        'num_subjects': n_subj,
        't_max_neuronal': t_max_neuronal,
        't_warmup': t_warmup,
        'sampling_period': sampling_period
    } for i, _ in enumerate(rn)]

    results = pool.map(compute_g, ee, gs)

if __name__ == '__main__':
    run()

