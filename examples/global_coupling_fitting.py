import argparse
import copy
import csv
import gc
import glob
import itertools
import os
import time
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from numpy.ma.core import repeat

import numba
# Disable JIT compilation for debugging purposes
# numba.config.DISABLE_JIT = True  

from neuronumba.bold import BoldStephan2008
from neuronumba.observables.sw_fcd import SwFCD
from neuronumba.simulator.models import Deco2014, ZerlautAdaptationFirstOrder, ZerlautAdaptationSecondOrder, Hopf, Montbrio
from neuronumba.fitting.fic.fic import FICDeco2014
from neuronumba.tools import filterps, hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import PhFCD, FC

# Local module
from neuronumba.observables.accumulators import ConcatenatingAccumulator, AveragingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic, PearsonSimilarity
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.tools.loader import load_2d_matrix


class ObservableConfig:
    """
    Configuration class for observables that encapsulates the observable,
    accumulator, distance measure, and optional band-pass filter.
    """
    
    def __init__(self, observable, accumulator, distance_measure, band_pass_filter=None):
        """
        Initialize observable configuration.
        
        Args:
            observable: The observable instance (FC, PhFCD, SwFCD, etc.)
            accumulator: The accumulator instance (AveragingAccumulator, ConcatenatingAccumulator)
            distance_measure: The distance measure instance (PearsonSimilarity, KolmogorovSmirnovStatistic)
            band_pass_filter: Optional band-pass filter instance (BandPassFilter or None)
        """
        self.observable = observable
        self.accumulator = accumulator
        self.distance_measure = distance_measure
        self.band_pass_filter = band_pass_filter
    
    def init_accumulator(self, num_subjects, n_rois):
        """Initialize the accumulator for this observable."""
        return self.accumulator.init(num_subjects, n_rois)
    
    def process_signal(self, signal, pos, measure_values):
        """
        Process a signal through the observable and accumulate results.
        
        Args:
            signal: The input signal
            pos: Position/index for accumulation
            measure_values: The accumulated measure values
            
        Returns:
            Updated measure values
        """
        # Apply band-pass filter if configured (create a copy to avoid modifying original)
        processed_signal = signal
        if self.band_pass_filter is not None:
            processed_signal = self.band_pass_filter.filter(signal.copy())
        
        # Process signal through observable
        observable_result = self.observable.from_fmri(processed_signal)
        
        # Get the observable name (key) for the processed signal
        observable_name = next(iter(observable_result.keys()))
        
        # Accumulate results
        return self.accumulator.accumulate(measure_values, pos, observable_result[observable_name])
    
    def postprocess(self, measure_values):
        """Postprocess the accumulated measure values."""
        return self.accumulator.postprocess(measure_values)
    
    def compute_distance(self, simulated_data, empirical_data):
        """Compute distance between simulated and empirical data."""
        return self.distance_measure.distance(simulated_data, empirical_data)
    
    def __repr__(self):
        return (f"ObservableConfig(observable={self.observable.__class__.__name__}, "
                f"accumulator={self.accumulator.__class__.__name__}, "
                f"distance_measure={self.distance_measure.__class__.__name__}, "
                f"band_pass_filter={self.band_pass_filter.__class__.__name__ if self.band_pass_filter else None})")


def create_observable_config(observable_name, distance_measure, band_pass_filter=None):
    """
    Factory function to create ObservableConfig instances.
    
    Args:
        observable_name: Name of the observable ('FC', 'phFCD', 'swFCD')
        distance_measure: Distance measure instance
        band_pass_filter: Optional band-pass filter
        
    Returns:
        ObservableConfig instance
    """
    if observable_name == 'FC':
        return ObservableConfig(FC(), AveragingAccumulator(), distance_measure, band_pass_filter)
    elif observable_name == 'phFCD':
        return ObservableConfig(PhFCD(), ConcatenatingAccumulator(), distance_measure, band_pass_filter)
    elif observable_name == 'swFCD':
        return ObservableConfig(SwFCD(), ConcatenatingAccumulator(), distance_measure, band_pass_filter)
    else:
        raise RuntimeError(f"Observable <{observable_name}> not supported!")


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


def load_subjects_data(fmri_path, max_subjects=21):
    # This function is highly dependent on your input data layout
    # This example is configured for the data downloaded by EBRAINS
    # Read: examples/Data_Raw/ebrains_popovych/README.md
    if not os.path.isdir(fmri_path):
        raise FileNotFoundError(f"Path <{fmri_path}> does not exist or is not a folder!")
    n_sub = 0
    result = {}
    for n_sub in range(max_subjects):
        subject_path = os.path.join(fmri_path, f'{n_sub:03d}')
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
    # observables is a dictionary of {observableName: ObservableConfig}
    num_subjects = len(bold_signals)
    N = bold_signals[next(iter(bold_signals))].shape[1]

    observables = exec_env['observables']

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds in observables:  # Initialize data structs for each observable
        measureValues[ds] = observables[ds].init_accumulator(num_subjects, N)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, bold_signals[s].shape[0], bold_signals[s].shape[1]), end='', flush=True)
        signal = bold_signals[s]  # LR_version_symm(tc[s])
        start_time = time.perf_counter()

        for ds in observables:  # Now, let's compute each measure and store the results
            observable_config = observables[ds]
            measureValues[ds] = observable_config.process_signal(signal, pos, measureValues[ds])

        print(" -> computed in {} seconds".format(time.perf_counter() - start_time))

    for ds in observables:  # finish computing each observable
        observable_config = observables[ds]
        measureValues[ds] = observable_config.postprocess(measureValues[ds])

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


def compute_g(exec_env, g):
    out_file = exec_env['out_file']
    force_recomputations = False if 'force_recomputations' not in exec_env else exec_env['force_recomputations']

    observables = exec_env['observables']
    processed = exec_env['processed']

    if not force_recomputations and os.path.exists(out_file):
        print(f"Loading previous data for g={g}")
        sim_measures = hdf.loadmat(out_file)
    else:
        print(f"Starting computation for g={g}")
        sim_measures = eval_one_param(exec_env, g)
        sim_measures['g'] = g  # Add the g value to the result
        for ds in observables:
            sim_measures[f'dist_{ds}'] = observables[ds].compute_distance(sim_measures[ds], processed[ds])
        hdf.savemat(out_file, sim_measures)

    for ds in observables:
        print(f" {ds} for g={g}: {sim_measures[f'dist_{ds}']};", end='', flush=True)

    return sim_measures


def process_empirical_subjects(bold_signals, observables: dict[str, ObservableConfig], verbose=True):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observables is a dictionary of {observableName: ObservableConfig}
    num_subjects = len(bold_signals)
    # get the first key to retrieve the value of N = number of areas
    n_rois = bold_signals[next(iter(bold_signals))].shape[0]

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds, observable_config in observables.items():
        measureValues[ds] = observable_config.init_accumulator(num_subjects, n_rois)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        # BOLD signals from file have inverse shape
        signal = bold_signals[s].T  # need to be transposed for the rest of NeuroNumba...

        print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, signal.shape[0],
                                                                      signal.shape[1]), flush=True)

        for ds, observable_config in observables.items():
            start_time = time.perf_counter()
            measureValues[ds] = observable_config.process_signal(signal, pos, measureValues[ds])
            if verbose:
                print(f"   Time to process observable {ds} for subject {s}: {time.perf_counter() - start_time:.2f} seconds")

    for ds, observable_config in observables.items():
        measureValues[ds] = observable_config.postprocess(measureValues[ds])

    return measureValues


def executor_simulate_single_subject(n, exec_env, g):
    try:
        return n, simulate_single_subject(exec_env, g)
    except Exception as e:
        print(f"Error simulating subject {n}: {e}", file=sys.stderr)
        raise


def compute_g_mp(exec_env, g, nproc):
    out_file = exec_env['out_file']

    if os.path.exists(out_file):
        print(f"File {out_file} already exists, skipping...")
        return    

    print(f'Computing distance for G={g}', flush=True)

    subjects = list(range(exec_env['num_subjects']))
    results = []
    print(f'Creating process pool with {nproc} workers')
    while len(subjects) > 0:
        print(f"EXECUTOR --- START cycle for {len(subjects)} subjects")
        pool = ProcessPoolExecutor(max_workers=nproc)
        futures = []
        future2subj = {}
        for n in subjects:
            f = pool.submit(executor_simulate_single_subject, n, exec_env, g)
            future2subj[f] = n
            futures.append(f)

        print(f"EXECUTOR --- WAITING for {len(futures)} futures to finish")

        subjects = []
        for future in as_completed(futures):
            try:
                n, result = future.result()
                results.append((n, result))
                print(f"EXECUTOR --- FINISHED subject {n}")
            except Exception as exc:
                print(f"EXECUTOR --- FAIL subject {n}. Restarting pool.")
                pool.shutdown(wait=False)
                finished = [n for n, _ in results]
                subjects = [n for n in subjects if n not in finished]
                break

    simulated_bolds = {}
    for n, r in results:
        simulated_bolds[n] = r
    

    sim_measures = process_bold_signals(simulated_bolds, exec_env)
    sim_measures['g'] = g
    processed = exec_env['processed']
    
    observables = exec_env['observables']
    for ds in observables:
        sim_measures[f'dist_{ds}'] = observables[ds].compute_distance(sim_measures[ds], processed[ds])

    hdf.savemat(out_file, sim_measures)


def run(args):

    # This is the data related to the dataset that we are going to load.
    # For your dataset, you need to figure these values

    # Time resolution parameter of the fMRI data (seconds). Each dataset will have its own tr value
    # depending on the scanner setting used to capture the BOLD signal
    if args.tr is None:
        raise RuntimeError("Please provide the --tr parameter with the time resolution of the fMRI scanner (milliseconds)")
    else:
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
        integrator = EulerStochastic(dt=dt, sigmas=np.r_[0.0, 0.0, 0.0, 0.0, 1e-3, 1e-3])
        bold = True
        obs_var = 'r_e'
    elif args.model == 'Zerlaut2O':
        model = ZerlautAdaptationSecondOrder()
        integrator = EulerStochastic(dt=dt, sigmas=np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-3])
        bold = True
        obs_var = 'E'
    else:
        raise RuntimeError(f"Model <{args.model}> not supported!")

    if args.measure == 'PS':
        measure = PearsonSimilarity()
    else:
        measure = KolmogorovSmirnovStatistic()

    bpf = BandPassFilter(tr=tr, k=args.bpf[0], flp=args.bpf[1], fhi=args.bpf[2]) if args.bpf is not None else None

    observables = {}
    for observable_name in args.observables:
        observables[observable_name] = create_observable_config(observable_name, measure, bpf)

    all_fMRI = {s: d for s, d in enumerate(timeseries)}

    out_file_path = os.path.join(args.out_path, f"sim_{args.model}_{'bpf' if bpf else 'nobpf'}")    
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)

    # Process (or load) empirical data
    emp_filename = os.path.join(out_file_path, 'fNeuro_emp.mat')
    if not os.path.exists(emp_filename):
        processed = process_empirical_subjects(all_fMRI, observables)
        hdf.savemat(emp_filename, processed, prev_73=True)
    else:
        processed = {observable_name: load_2d_matrix(emp_filename, index=observable_name) for observable_name in observables.keys()}

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_g{}.mat')

    n_subj = args.nsubj if args.nsubj is not None else n_frmis

    if args.plot_g:
        # Show distances for all G files generated
        file_list = glob.glob(os.path.join(out_file_path, 'fitting_g*.mat'))
        y = {}
        for o_name in observables.keys():
            y[o_name] = []
        x = []
        for f in sorted(file_list):
            m = hdf.loadmat(f)
            g = m['g']
            for o_name in y.keys():
                d = m[f'dist_{o_name}']
                print(f"Distance for g={g} and observable {o_name} = {d}", flush=True)
                y[o_name].append(d)
            x.append(g)

        for o_name, ys in y.items():
            fig, ax = plt.subplots()
            ax.plot(x, ys)
            ax.set_title(f"Distance for observable {o_name}")
            plt.savefig(os.path.join(out_file_path, f"fig_g_optim_{o_name}.png"), dpi=300)

    elif args.g is not None and not args.use_mp:
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
            'out_file': out_file_name_pattern.format(np.round(args.g, decimals=3)),
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
            'force_recomputations': False,
        }, args.g)

    elif args.g is not None and args.use_mp:
        compute_g_mp({
            'verbose': True,
            'model': copy.deepcopy(model),
            'integrator': copy.deepcopy(integrator),
            'weights': sc_norm,
            'processed': processed,
            'tr': tr,
            'observables': copy.deepcopy(observables),
            'obs_var': obs_var,
            'bold': bold,
            'bold_model': BoldStephan2008().configure(),
            'out_file': out_file_name_pattern.format(np.round(args.g, decimals=3)),
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period
        }, args.g, args.nproc)

    elif args.g_range is not None:
        [g_Start, g_End, g_Step] = args.g_range
        gs = np.arange(g_Start, g_End + g_Step, g_Step)

        results = []
        remaining_gs = list(gs)
        finished_gs = []
        
        while len(remaining_gs) > 0:
            print(f'Creating process pool with {args.nproc} workers')
            pool = ProcessPoolExecutor(max_workers=args.nproc)
            futures = []
            future_to_g = {}
            
            print(f"EXECUTOR --- START cycle for {len(remaining_gs)} gs")
            for gf in remaining_gs:
                exec_env = {
                    'verbose': True,
                    'model': copy.deepcopy(model),
                    'integrator': copy.deepcopy(integrator),
                    'weights': sc_norm,
                    'processed': processed,
                    'tr': tr,
                    'observables': copy.deepcopy(observables),
                    'obs_var': obs_var,
                    'bold': bold,
                    'bold_model': BoldStephan2008().configure(),
                    'out_file': out_file_name_pattern.format(np.round(gf, decimals=3)),
                    'num_subjects': n_subj,
                    't_max_neuronal': t_max_neuronal,
                    't_warmup': t_warmup,
                    'sampling_period': sampling_period
                }
                future = pool.submit(compute_g, exec_env, gf)
                future_to_g[future] = gf
                futures.append(future)

            print(f"EXECUTOR --- WAITING for {len(futures)} futures to finish")
            
            remaining_gs = []
            for future in as_completed(futures):
                gf = future_to_g[future]
                try:
                    result = future.result()
                    results.append(result)
                    finished_gs.append(gf)
                    print(f"EXECUTOR --- FINISHED process for g={gf}")
                except Exception as exc:
                    print(f"EXECUTOR --- FAIL computation for g={gf}. Error: {exc}")
                    remaining_gs = [g for g in gs if g not in finished_gs]

            pool.shutdown(wait=False, cancel_futures=True)

    else:
        RuntimeError("Neither --g not --g-range has been defined")


def gen_arg_parser():
    parser = argparse.ArgumentParser(description="Global coupling fitting script for NeuroNumba models.")
    parser.add_argument("--full-scan", action='store_true', default=False, help="Full scan all models/observables/measures")
    parser.add_argument("--verbose", action='store_true', default=False, help="Prints extra information during execution")
    parser.add_argument("--use-mp", action='store_true', default=False, help="Use multiprocessing if possible")
    parser.add_argument("--nproc", type=int, default=10, help="Number of parallel processes")
    parser.add_argument("--nsubj", type=int, help="Number of subjects for the simulations")
    parser.add_argument("--g", type=float, help="Single point execution for a global coupling value")
    parser.add_argument("--g-range", nargs=3, type=float, help="Parameter sweep range for G (start, end, step)")
    parser.add_argument("--bpf", nargs=3, type=float, required=False, help="Band pass filter to apply to BOLD signal (k, lp freq, hp freq)")
    parser.add_argument("--model", type=str, default='Deco2014', help="Model to use (Hopf, Deco2014, Montbrio, Zerlaut1O, Zerlaut2O)")
    parser.add_argument("--observables", nargs='+', type=str, required=True, help="Observables to use (FC, phFCD, swFCD)")
    parser.add_argument("--measure", type=str, default='PS', help="Measure to use (PearsonSimilarity (PS), KolmogorovStatistic (KS))")
    parser.add_argument("--out-path", type=str, required=True, help="Path to folder for output results")
    parser.add_argument("--tr", type=float, help="Time resolution of fMRI scanner (seconds)")
    parser.add_argument("--sc-scaling", type=float, default=0.2, help="Scaling factor for the SC matrix")
    parser.add_argument("--tmax", type=float, required=False, help="Override simulation time (seconds)")
    parser.add_argument("--fmri-path", type=str, help="Path to fMRI timeseries data")
    parser.add_argument("--plot-g", action='store_true', default=False, help="Plot G optimization results")

    return parser

if __name__ == '__main__':
    parser = gen_arg_parser()
    args = parser.parse_args()  # for example, for a single test, use --ge-range 1.0 10.0 1.0
    if args.full_scan:
        models = ['Deco2014', 'Hopf', 'Montbrio', 'Zerlaut2O']
        observables = ['FC', 'phFCD', 'swFCD']
        measures = ['PS', 'KS']
        args = [sys.argv[0], '--nproc', '5', '--g-range', '1', '20', '0.2', '--tr', '720', '--tmax', '600', '--fmri-path', './Data_Raw/ebrains_popovych', '--out-path', './Data_Produced/ebrains_popovych']
        for model, observable, measure in list(itertools.product(models, observables, measures)):
            sys.argv = args + ['--model', model, '--observable', observable, '--measure', measure]
            print(f'Running fitting for model {model}, observable {observable}, measure {measure}')
            run()
    else:
        run(args)

