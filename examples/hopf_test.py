import argparse
import csv
import os
import random
import time

import h5py
import numpy as np
from pathos.multiprocessing import ProcessPool

from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.models import Deco2014
from neuronumba.tools import filterps, hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import PhFCD

# Local module
import plot
from neuronumba.observables.accumulators import ConcatenatingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.models.hopf import Hopf
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator


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


def load_subjects_data(fMRI_path, out_file_path, num_sample_subjects):
    fMRIs = read_matlab_h5py(fMRI_path)
    selected_subjects_file = os.path.join(out_file_path, 'list_ids.txt')
    # ---------------- fix subset of subjects to sample
    if not os.path.isfile(selected_subjects_file):  # if we did not already select a list...
        list_ids = random.sample(sorted(fMRIs.keys()), num_sample_subjects)
        save_selected_subjcets(selected_subjects_file, list_ids)
    else:  # if we did, load it!
        list_ids = load_subject_list(selected_subjects_file)
    # ---------------- OK, let's proceed
    nNodes, Tmax = fMRIs[next(iter(fMRIs))].shape
    res = np.zeros((num_sample_subjects, nNodes, Tmax))
    for pos, s in enumerate(list_ids):
        res[pos] = fMRIs[s]
    return res, list_ids


def simulate(m, weights, we, obs_var, bold=False):
    m.configure(g=we)
    n_rois = weights.shape[0]
    sampling_period = 1.0
    lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
    speed = 1.0
    con = Connectivity(weights=weights, lengths=lengths, speed=speed)
    dt = 0.1
    # integ = EulerDeterministic(dt=dt)
    integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2])

    # coupling = CouplingLinearDense(weights=weights, delays=con.delays, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois)
    history = HistoryNoDelays()
    # mnt = TemporalAverage(period=1.0, dt=dt)
    monitor = RawSubSample(period=sampling_period, state_vars=m.get_state_sub([obs_var]), obs_vars=m.get_observed_sub())
    s = Simulator(connectivity=con, model=m, history=history, integrator=integ, monitors=[monitor])
    start_time = time.perf_counter()
    s.run(0, t_max_neuronal + t_warmup)
    t_sim = time.perf_counter() - start_time
    data = monitor.data(obs_var)
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(data.shape[0]), data)
    # plt.show()
    data_from = int(data.shape[0] * t_warmup / (t_max_neuronal + t_warmup))
    signal = data[data_from:, :]
    if bold:
        b = BoldStephan2008()
        signal = b.compute_bold(signal, monitor.period)
    return t_sim, signal


def simulate_single_subject(m, weights, we, obs_var, bold=False):
    t, signal = simulate(m, weights, we, obs_var, bold)
    n_min = int(np.round(t_min / dtt))
    step = int(np.round(tr/dtt))
    bds = signal[n_min+step-1::step, :]
    return bds


def process_bold_signals(bold_signals, observables):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    num_subjects = len(bold_signals)
    N = bold_signals[next(iter(bold_signals))].shape[1]

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


def eval_one_param(m, weights, we, obs_var, observables, num_subjects, bold=False):
    simulated_bolds = {}
    start_time = time.perf_counter()
    for nsub in range(num_subjects):  # trials. Originally it was 20.
        print(f"   Simulating we={we} -> subject {nsub}/{num_subjects}!!!")
        bds = simulate_single_subject(m, weights, we, obs_var,bold)
        repetitionsCounter = 0
        while np.isnan(bds).any() or (np.abs(bds) > np.inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
            raise RuntimeError
        simulated_bolds[nsub] = bds

    dist = process_bold_signals(simulated_bolds, observables)
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist['we'] = we

    return dist


def compute_we(m, num_sim_subjects, obs_var, observables, out_file_name_pattern, processed, we, weights, bold=False):
    result = {}
    out_file = out_file_name_pattern.format(np.round(we, decimals=3))
    if os.path.exists(out_file):
        print(f"Loading previous data for we={we}")
        sim_measures = hdf.loadmat(out_file)
    else:
        print(f"Starting computation for we={we}")
        sim_measures = eval_one_param(m, weights, we, obs_var, observables, num_sim_subjects, bold)
        hdf.savemat(out_file, sim_measures)
    for ds in observables:
        result[ds] = observables[ds][2].distance(sim_measures[ds], processed[ds])

        print(f" {ds} for we={we}: {result[ds]};", end='', flush=True)

    return result


def preprocessing_pipeline(out_file_path, m, weights, processed,  #, abeta,
                           observables,  # This is a dictionary of {name: (distance module, apply filters bool)}
                           wes, bold=False):
    print("\n\n###################################################################")
    print("# Compute ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    obs_var = 'x'
    num_parms = len(wes)

    num_sim_subjects = 20

    fitting = {}
    for ds in observables:
        fitting[ds] = np.zeros((num_parms))

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_we{}.mat')

    pool = ProcessPool(nodes=10)
    rn = list(range(len(wes)))
    ns = [num_sim_subjects for _ in rn]
    ov = [obs_var for _ in rn]
    ob = [observables for _ in rn]
    of = [out_file_name_pattern for _ in rn]
    pr = [processed for _ in rn]
    wt = [weights for _ in rn]
    b = [bold for _ in rn]
    ms = [m for _ in rn]
    results = pool.map(compute_we, ms, ns, ov, ob, of, pr, wes, wt, b)

    return results


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--we-range", nargs=3, help="Parameter sweep range for G", type=float, required=True)
    parser.add_argument("--model", help="Model to use (Hopf, Deco2014)", type=str, default='Hopf')
    parser.add_argument("--out-path", help="Path to folder for output results", type=str, required=True)
    parser.add_argument("--sc-matrix", help="Path to SC matrix (Matlab file)", type=str, required=True)
    parser.add_argument("--fmri-path", help="Path to fMRI timeseries data (Matlab file)", type=str, required=True)

    args = parser.parse_args()  # for example, for a single test, use --we-range 1.0 1.1 1.0

    [wStart, wEnd, wStep] = args.we_range

    numSampleSubjects = 20
    tr = 2.0
    dtt = 2.0
    t_min = 10.0 * tr
    sampling_period = tr
    warmup_factor = 606.0/2000.0

    timeseries, listIDs = load_subjects_data(args.fmri_path, args.out_path, numSampleSubjects)
    n_subj, n_rois, t_max = timeseries.shape
    t_max_neuronal = (t_max - 1) * tr + 30
    t_warmup = t_max_neuronal/warmup_factor

    # -------------------------- Load SC matrix
    mat0 = hdf.loadmat(args.sc_matrix)['SC_dbs80FULL']
    sc_norm = 0.2 * mat0 / mat0.max()

    bold = False
    if args.model == 'Hopf':
        # -------------------------- Setup Hopf frequencies
        bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=tr, apply_detrend=True, apply_demean=True)
        f_diff = filterps.filt_pow_spetra_multiple_subjects(timeseries, tr, bpf)  # baseline_group[0].reshape((1,52,193))
        omega = 2 * np.pi * f_diff
        model = Hopf(omega=omega, weights=sc_norm, a=-0.02)
    elif args.model == 'Deco2014':
        model = Deco2014()
        bold = True
    else:
        raise RuntimeError(f"Model <{args.model}> not supported!")

    # -------------------------- Observable definition
    observables = {'phFCD': (PhFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic())}

    all_fMRI = {s: d for s, d in enumerate(timeseries)}

    # -------------------------- Process (or load) empirical data
    emp_filename = os.path.join(args.out_path, 'fNeuro_emp.mat')
    if os.path.exists(emp_filename):
        processed = hdf.loadmat(emp_filename)
    else:
        processed = process_empirical_subjects(all_fMRI, observables, bpf)
        hdf.savemat(emp_filename, processed)


    # -------------------------- Preprocessing pipeline!!!
    wes = np.arange(wStart, wEnd + wStep, wStep)
    optimal = preprocessing_pipeline(args.out_path, model, sc_norm, processed, observables, wes, bold)
    # =======  Only for quick load'n plot test...
    plot.load_and_plot(os.path.join(args.out_path, 'fitting_we{}.mat'), observables,
                              WEs=wes, weName='we',
                              empFilePath=os.path.join(args.out_path, 'fNeuro_emp.mat'))

