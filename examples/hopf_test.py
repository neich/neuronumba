import argparse
import csv
import os
import random
import time

import h5py
import numpy as np
from matplotlib import pyplot as plt

from neuronumba.bold import BoldStephan2008
from neuronumba.numba_tools import hdf
from neuronumba.tools import filterps
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import PhFCD

# Local module
import plot
from neuronumba.observables.accumulators import ConcatenatingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.coupling import CouplingNoDelays
from neuronumba.simulator.integrators.euler import EulerDeterministic, EulerStochastic
from neuronumba.simulator.models.hopf import Hopf
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator

tr = 2.0
dtt = 2.0
t_min = 10.0 * tr
sampling_period = tr
warmup_factor = 606.0/2000.0
t_max_neuronal = None
t_warmup = None

# Hopf parameters
omega = None
a = -0.02

# Here you shoud use the path to your data
cwd = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(cwd, 'Data_Raw', 'HCP', 'DataHCP80')
# Empirical data with the BOLD signal for a set of subjects
fMRI_rest_path = os.path.join(base_path, 'hcp1003_REST_LR_dbs80.mat')
# Connectivity matrix
SC_path = os.path.join(base_path, 'SC_dbs80HARDIFULL.mat')

out_file_path = os.path.join(cwd, 'Data_Produced', 'Tests', 'TestHopf')
# How many of the original subjects l
numSampleSubjects = 20
selected_subjects_file = os.path.join(out_file_path, f'selected_{numSampleSubjects}.txt')


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


def load_subjects_data(fMRI_path, num_sample_subjects):
    fMRIs = read_matlab_h5py(fMRI_path)
    # ---------------- fix subset of subjects to sample
    if not os.path.isfile(selected_subjects_file):  # if we did not already select a list...
        list_ids = random.sample(fMRIs.keys(), num_sample_subjects)
        save_selected_subjcets(selected_subjects_file, list_ids)
    else:  # if we did, load it!
        list_ids = load_subject_list(selected_subjects_file)
    # ---------------- OK, let's proceed
    nNodes, Tmax = fMRIs[next(iter(fMRIs))].shape
    res = np.zeros((num_sample_subjects, nNodes, Tmax))
    for pos, s in enumerate(list_ids):
        res[pos] = fMRIs[s]
    return res, list_ids


def sim_hopf(weights, we, obs_var):
    n_rois = weights.shape[0]
    lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
    speed = 1.0
    con = Connectivity(weights=weights, lengths=lengths, speed=speed)
    m = Hopf(a=a, omega=omega, g=we, weights=weights)
    dt = 0.1
    # integ = EulerDeterministic(dt=dt)
    integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 1e-3])

    # coupling = CouplingLinearDense(weights=weights, delays=con.delays, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois)
    coupling = CouplingNoDelays(weights=weights)
    # mnt = TemporalAverage(period=1.0, dt=dt)
    monitor = RawSubSample(period=sampling_period, state_vars=m.get_state_sub([obs_var]), obs_vars=m.get_observed_sub())
    s = Simulator(connectivity=con, model=m, coupling=coupling, integrator=integ, monitors=[monitor])
    start_time = time.perf_counter()
    s.run(0, t_max_neuronal + t_warmup)
    t_sim = time.perf_counter() - start_time
    data = monitor.data(obs_var)
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(data.shape[0]), data)
    # plt.show()
    data_from = int(data.shape[0] * t_warmup / (t_max_neuronal + t_warmup))
    return t_sim, data[data_from:, :]


def simulate_single_subject(weights, we, obs_var):
    t, signal = sim_hopf(weights, we, obs_var)
    n_min = int(np.round(t_min / dtt))
    step = int(np.round(tr/dtt))
    # No need for a BOLD simulation, the result of the model directly can be used as BOLD signal
    # Discard the first n_min samples
    bds = signal[n_min+step-1::step, :]
    return bds


def process_bold_signals(bold_signals, observables):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    num_subjects = len(bold_signals)
    N = bold_signals[next(iter(bold_signals))].shape[0]  # get the first key to retrieve the value of N = number of areas

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
            bold_filt = bpf.filter(signal.T)
            # FC, swFCD, phFCD, ...
            proc_signal = measure.from_fmri(bold_filt)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, proc_signal[ds])

        print(" -> computed in {} seconds".format(time.perf_counter() - start_time))

    for ds in observables:  # finish computing each observable
        accumulator = observables[ds][1]  # FC, swFCD, phFCD, ...
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def eval_one_param(weights, we, obs_var, observables, num_subjects):
    simulated_bolds = {}
    start_time = time.perf_counter()
    for nsub in range(num_subjects):  # trials. Originally it was 20.
        print(f"   Simulating we={we} -> subject {nsub}/{num_subjects}!!!")
        bds = simulate_single_subject(weights, we, obs_var)
        repetitionsCounter = 0
        while np.isnan(bds).any() or (np.abs(bds) > np.inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
            raise RuntimeError
        simulated_bolds[nsub] = bds

    dist = process_bold_signals(simulated_bolds, observables)
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist['we'] = we

    return dist


def preprocessing_pipeline(weights, processed,  #, abeta,
                           observables,  # This is a dictionary of {name: (distance module, apply filters bool)}
                           wes):
    print("\n\n###################################################################")
    print("# Compute ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    balanced_parms = [{'we': we} for we in wes]
    obs_var = 'x'
    num_parms = len(wes)

    num_sim_subjects = 20

    fitting = {}
    for ds in observables:
        fitting[ds] = np.zeros((num_parms))

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_we{}.mat')
    for pos, we in enumerate(wes):
        print(f'Staring computation for we={we}')
        out_file = out_file_name_pattern.format(np.round(we, decimals=3))
        if os.path.exists(out_file):
            sim_measures = hdf.loadmat(out_file)
        else:
            sim_measures = eval_one_param(weights, we, obs_var, observables, num_sim_subjects)
            hdf.savemat(out_file, sim_measures)

        for ds in observables:
            fitting[ds][pos] = observables[ds][2].distance(sim_measures[ds], processed[ds])
            print(f" {ds}: {fitting[ds][pos]};", end='', flush=True)


    optimal = {}
    for sd in observables:
        optim = observables[sd][0].findMinMax(fitting[sd])
        optimal[sd] = (optim[0], optim[1], balanced_parms[optim[1]])
    return optimal


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
        signal = bold_signals[s]  # LR_version_symm(tc[s])

        signal_filt = bpf.filter(signal)
        for ds, (observable, accumulator, _) in observables.items():
            procSignal = observable.from_fmri(signal_filt)
            measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal[ds])

    for ds, (observable, accumulator, _) in observables.items():
        measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--we-range", nargs=3, help="Parameter sweep range for G", type=float, required=False)

    args = parser.parse_args()

    [wStart, wEnd, wStep] = args.we_range

    timeseries, listIDs = load_subjects_data(fMRI_rest_path, numSampleSubjects)
    n_subj, n_rois, t_max = timeseries.shape
    t_max_neuronal = (t_max - 1) * tr + 30
    t_warmup = t_max_neuronal/warmup_factor

    bpf = BandPassFilter(k=2, flp=0.008, fhi=0.08, tr=2.0, apply_detrend=True, apply_demean=True)
    f_diff = filterps.filtPowSpetraMultipleSubjects(timeseries, tr, bpf)  # baseline_group[0].reshape((1,52,193))
    # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
    # f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])
    omega = 2 * np.pi * f_diff


    all_fMRI = {s: d for s, d in enumerate(timeseries)}

    emp_filename = os.path.join(out_file_path, 'fNeuro_emp.mat')
    if os.path.exists(emp_filename):
        processed = hdf.loadmat(emp_filename)
    else:
        observables = {'phFCD': (PhFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic())}
        processed = process_empirical_subjects(all_fMRI, observables, bpf)
        hdf.savemat(emp_filename, processed)


    observables = {'phFCD': (PhFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic())}

    mat0 = hdf.loadmat(SC_path)['SC_dbs80FULL']
    sc_norm = 0.2 * mat0 / mat0.max()

    wes = np.arange(wStart, wEnd + wStep, wStep)
    optimal = preprocessing_pipeline(sc_norm, processed, observables, wes)
    # =======  Only for quick load'n plot test...
    plot.load_and_plot(out_file_path + '/fitting_we{}.mat', observables,
                              WEs=wes, weName='we',
                              empFilePath=out_file_path+'/fNeuro_emp.mat')

    print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    print("".join(f" - Optimal {k}({optimal[k][2]})={optimal[k][0]}\n" for k in optimal))
