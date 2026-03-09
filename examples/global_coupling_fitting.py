import argparse
import ast
import copy
import csv
import gc
import glob
import itertools
import os
import secrets
import time

from typing import Callable, TypedDict

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# import numba
# Disable JIT compilation for debugging purposes
# numba.config.DISABLE_JIT = True  

from neuronumba.bold import BoldStephan2008, BoldStephan2007, BoldStephan2007Alt
from neuronumba.observables.sw_fcd import SwFCD
from neuronumba.simulator.models import Deco2014, ZerlautAdaptationSecondOrder, Hopf, Montbrio
from neuronumba.fitting.fic.fic import FICDeco2014
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import PhFCD, FC

# Local module
from neuronumba.observables.accumulators import ConcatenatingAccumulator, AveragingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic, PearsonSimilarity
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.tools.loader import load_2d_matrix
from neuronumba.tools.random import set_seed


class ExecEnv(TypedDict, total=False):
    """
    Execution environment dictionary for simulation runs.

    Carries all configuration needed by compute_g / compute_g_mp and the
    functions they call.  All fields are optional (total=False) because
    different code paths require different subsets.

    Attributes:
        verbose: Print progress and timing information during execution.
        model: Registered model name (e.g. 'Deco2014', 'Hopf'). Must exist in ModelFactory.
        model_attributes: Extra attributes passed to model.set_attributes() (e.g. {'auto_fic': True}).
        dt: Integration time step in milliseconds.
        weights: Structural connectivity matrix, shape (n_rois, n_rois).
        processed: Pre-computed empirical observables, keyed by observable name.
        tr: Repetition time of the fMRI scanner in milliseconds.
        observables: Observable specs as passed on the CLI (e.g. ['FC,PS', 'swFCD,KS']).
        bpf: Band-pass filter applied to BOLD signals before computing observables.
        obs_var: Name of the model variable to record (must be in state_vars or observable_vars).
        bold: Whether to generate BOLD signal from the raw neuronal signal.
        bold_model: Configured BOLD model instance (from BoldModelFactory).
        out_file: Path where simulation results are saved (.mat).
        num_subjects: Number of independent simulation trials (virtual subjects).
        t_max_neuronal: Total neuronal simulation time in milliseconds (excluding warmup).
        t_warmup: Warmup period in milliseconds (discarded from output).
        sampling_period: Raw signal sampling period in milliseconds.
        force_recomputations: If True, ignore cached results and recompute.
        scale_signal: Multiplicative factor applied to the raw signal before BOLD conversion.
        J: FIC balance vector, shape (n_rois,). Set automatically when J_file_name_pattern is used.
        J_file_name_pattern: Format string for FIC files (e.g. 'path/J_{}.mat'). When present,
            FIC is loaded or computed and stored in 'J' before simulation.
        weights_sigma_factor: When > 0, adds Gaussian noise to the SC matrix scaled by this factor
            times the matrix maximum, producing a perturbed copy per simulation call.
        callback_simulate_single_subject: Optional callable(n, exec_env, signal, bold) invoked
            after each subject in the multiprocessing executor path.
    """
    verbose: bool
    model: str
    model_attributes: dict
    dt: float
    weights: np.ndarray
    processed: dict
    tr: float
    observables: list[str]
    bpf: BandPassFilter | None
    obs_var: str
    bold: bool
    bold_model: object
    out_file: str
    num_subjects: int
    t_max_neuronal: float
    t_warmup: float
    sampling_period: float
    force_recomputations: bool
    scale_signal: float
    J: np.ndarray
    J_file_name_pattern: str
    weights_sigma_factor: float
    callback_simulate_single_subject: Callable[[int, 'ExecEnv', np.ndarray, np.ndarray | None], None]



def get_model_attributes(exec_env: ExecEnv):
    return exec_env['model_attributes'] if 'model_attributes' in exec_env and exec_env['model_attributes'] else {}


class ObservableConfig:
    """
    Configuration class for observables that encapsulates the observable,
    accumulator, distance measure, and optional band-pass filter.
    """
    
    def __init__(self, observable, accumulator, distance_measures, band_pass_filter=None):
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
        self.distance_measures = distance_measures
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
        return { dname: dist.distance(simulated_data, empirical_data) for dname, dist in self.distance_measures.items() }

    def __repr__(self):
        return (f"ObservableConfig(observable={self.observable.__class__.__name__}, "
                f"accumulator={self.accumulator.__class__.__name__}, "
                f"distance_measures={self.distance_measures.__class__.__name__}, "
                f"band_pass_filter={self.band_pass_filter.__class__.__name__ if self.band_pass_filter else None})")


def create_observable_config(observable_name, distance_measures, band_pass_filter=None):
    """
    Factory function to create ObservableConfig instances.
    
    Args:
        observable_name: Name of the observable ('FC', 'phFCD', 'swFCD')
        distance_measure: Distance measure name (PS or KS)
        band_pass_filter: Optional band-pass filter
        
    Returns:
        ObservableConfig instance
    """
    measures = {}
    for d in distance_measures:
        if d == 'PS':
            measures[d] = PearsonSimilarity()
        elif d == 'KS':
            measures[d] = KolmogorovSmirnovStatistic()
        else:
            raise ValueError(f"Unknown distance measure: {d}")
    if observable_name == 'FC':
        return ObservableConfig(FC(), AveragingAccumulator(), measures, band_pass_filter)
    elif observable_name == 'phFCD':
        return ObservableConfig(PhFCD(), ConcatenatingAccumulator(), measures, band_pass_filter)
    elif observable_name == 'swFCD':
        return ObservableConfig(SwFCD(), ConcatenatingAccumulator(), measures, band_pass_filter)
    else:
        raise RuntimeError(f"Observable <{observable_name}> not supported!")


def create_observables_dict(observables, bpf):
    obs_dict = {}
    for item in observables:
        l = item.split(",")
        obs_name = l[0]
        if obs_name in obs_dict:
            raise ValueError(f"Duplicate observable name: {obs_name}")
        measures = l[1:]
        obs_dict[obs_name] = create_observable_config(obs_name, measures, bpf)
    return obs_dict


class ModelFactory:
    _creators = {
        'Montbrio': lambda: Montbrio(),
        'Deco2014': lambda: Deco2014(),
        'Zerlaut2O': lambda: ZerlautAdaptationSecondOrder(),
        'Hopf': lambda: Hopf()
    }

    @staticmethod
    def create_model(model_name):
        if model_name not in ModelFactory._creators:
            raise ValueError(f"Unknown model name: {model_name}")
        return ModelFactory._creators[model_name]()

    @staticmethod
    def list_available_models():
        return list(ModelFactory._creators.keys())
    
    @staticmethod
    def add_model(model_name, creator):
        ModelFactory._creators[model_name] = creator


class BoldModelFactory:
    """Factory for creating BOLD signal models."""
    
    _creators = {
        'Stephan2008': lambda: BoldStephan2008().configure(),
        'Stephan2007': lambda: BoldStephan2007().configure(),
        'Stephan2007Alt': lambda: BoldStephan2007Alt().configure(),
    }

    @staticmethod
    def create_model(model_name):
        """
        Create a BOLD model instance.
        
        Args:
            model_name: Name of the BOLD model ('Stephan2008', 'Stephan2007', 'Stephan2007Alt')
            
        Returns:
            Configured BOLD model instance
        """
        if model_name not in BoldModelFactory._creators:
            raise ValueError(f"Unknown BOLD model: {model_name}. Available: {BoldModelFactory.list_available_models()}")
        return BoldModelFactory._creators[model_name]()

    @staticmethod
    def list_available_models():
        """List all available BOLD model names."""
        return list(BoldModelFactory._creators.keys())
    
    @staticmethod
    def add_model(model_name, creator):
        """
        Add a new BOLD model configuration.
        
        Args:
            model_name: Name of the BOLD model
            creator: Function that returns a configured BOLD model instance
        """
        BoldModelFactory._creators[model_name] = creator


class IntegratorFactory:
    """Factory for creating integrators with appropriate noise configurations for different models."""
    
    _configurations = {
        'Hopf': lambda dt: EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2]),
        'Deco2014': lambda dt: EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2]),
        'Montbrio': lambda dt: EulerStochastic(dt=dt, sigmas=np.r_[0.0, 0.0, 0.0, 0.0, 1e-3, 1e-3]),
        'Zerlaut2O': lambda dt: EulerStochastic(dt=dt, sigmas=np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-3]),
    }
    
    @staticmethod
    def create_integrator(model_name, dt):
        """
        Create an integrator appropriate for the given model.
        
        Args:
            model_name: Name of the model ('Hopf', 'Deco2014', 'Montbrio', 'Zerlaut2O')
            dt: Integration time step (ms)
            
        Returns:
            Configured integrator instance
        """
        if model_name not in IntegratorFactory._configurations:
            raise ValueError(f"Unknown model name for integrator: {model_name}")
        return IntegratorFactory._configurations[model_name](dt)
    
    @staticmethod
    def list_available_integrators():
        """List all available integrator configurations."""
        return list(IntegratorFactory._configurations.keys())
    
    @staticmethod
    def add_integrator_config(model_name, creator):
        """
        Add a new integrator configuration.
        
        Args:
            model_name: Name of the model
            creator: Function that takes dt and returns an integrator instance
        """
        IntegratorFactory._configurations[model_name] = creator
    
    @staticmethod
    def get_sigma_config(model_name):
        """Get the sigma configuration for a given model (for debugging/inspection)."""
        if model_name == 'Hopf':
            return np.r_[1e-2, 1e-2]
        elif model_name == 'Deco2014':
            return np.r_[1e-3, 1e-3]
        elif model_name == 'Montbrio':
            return np.r_[0.0, 0.0, 0.0, 0.0, 1e-3, 1e-3]
        elif model_name == 'Zerlaut2O':
            return np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-3]
        else:
            raise ValueError(f"Unknown model name: {model_name}")


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
    result = {}
    for n in range(max_subjects):
        subject_path = os.path.join(fmri_path, f'{n:03d}')
        if os.path.isdir(subject_path):
            fmri_file = os.path.join(subject_path, 'rfMRI_REST1_LR_BOLD.csv')
            if not os.path.isfile(fmri_file):
                raise FileNotFoundError(f"fMRI file <{fmri_file}> not found!")
            # We want the shape of each fmri to be (n_rois, t_max)
            result[n] = load_2d_matrix(fmri_file).T

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


def simulate(exec_env: ExecEnv, g):
    model = ModelFactory.create_model(exec_env['model']).set_attributes(get_model_attributes(exec_env))
    weights = exec_env['weights']
    if 'weights_sigma_factor' in exec_env:
        mmax = np.max(weights)
        weights = weights + exec_env['weights_sigma_factor'] * mmax * np.random.normal(size=exec_env['weights'].shape)
        weights = np.where(weights < 0, 0, weights)
        w = mmax / np.max(weights)
        weights /= w
    obs_var = exec_env['obs_var']
    t_max_neuronal = exec_env['t_max_neuronal']
    t_warmup = exec_env['t_warmup']
    integrator = IntegratorFactory.create_integrator(exec_env['model'], exec_env['dt'])
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


def simulate_single_subject(exec_env: ExecEnv, g):
    signal = simulate(exec_env, g) * exec_env.get('scale_signal', 1.0)
    sampling_period = exec_env['sampling_period']
    if exec_env['bold']:
        b = exec_env['bold_model']
        bds = b.compute_bold(signal, sampling_period)
        return signal, bds
    else:
        return signal, None

def _process_signals(signals, observables, verbose=True, transpose=False):
    """
    Core signal processing loop shared by process_bold_signals and process_empirical_subjects.

    Args:
        signals: Dictionary of subjects {key: signal_array}
        observables: Dictionary of {name: ObservableConfig}
        verbose: Whether to print progress information
        transpose: If True, transpose each signal before processing (for empirical data)

    Returns:
        Dictionary containing computed observables for all subjects
    """
    num_subjects = len(signals)
    first_signal = signals[next(iter(signals))]
    n_rois = first_signal.shape[0] if transpose else first_signal.shape[1]

    measure_values = {}
    for ds, obs_config in observables.items():
        measure_values[ds] = obs_config.init_accumulator(num_subjects, n_rois)

    for pos, s in enumerate(signals):
        signal = signals[s].T if transpose else signals[s]
        if verbose:
            print(f'   Processing signal {pos + 1}/{num_subjects} Subject: {s} ({signal.shape[0]}x{signal.shape[1]})',
                  end='', flush=True)
        start_time = time.perf_counter()

        for ds, obs_config in observables.items():
            measure_values[ds] = obs_config.process_signal(signal, pos, measure_values[ds])

        if verbose:
            print(f" -> {time.perf_counter() - start_time:.2f}s")

    for ds, obs_config in observables.items():
        measure_values[ds] = obs_config.postprocess(measure_values[ds])

    return measure_values


def process_bold_signals(bold_signals, observables, band_pass_filter=None, verbose=True):
    """
    Process BOLD signals and compute observables.

    Args:
        bold_signals: Dictionary of subjects {subjectName: subjectBOLDSignal}
        observables: Either a dict of ObservableConfig or a list of observable specs (e.g., ['FC,PS,KS'])
        band_pass_filter: Optional band-pass filter (only used when observables is a list)
        verbose: Whether to print progress information

    Returns:
        Dictionary containing computed observables for all subjects
    """
    if not isinstance(observables, dict):
        observables = create_observables_dict(observables, band_pass_filter)
    return _process_signals(bold_signals, observables, verbose, transpose=False)


def eval_one_param(exec_env: ExecEnv, g):
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
        _, bds = simulate_single_subject(exec_env, g)
        if np.isnan(bds).any() or np.isinf(bds).any():
            raise RuntimeError(f"Numeric error computing subject {nsub}/{num_subjects} for g={g}")
        simulated_bolds[nsub] = bds
        gc.collect()

    dist = process_bold_signals(simulated_bolds, _get_observables_dict(exec_env), verbose=exec_env.get('verbose', True))
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist['g'] = g

    return dist


def _get_observables_dict(exec_env: ExecEnv):
    """Get or create the observables dict, caching it in exec_env to avoid repeated creation."""
    if '_observables_dict' not in exec_env:
        exec_env['_observables_dict'] = create_observables_dict(exec_env['observables'], exec_env.get('bpf', None))
    return exec_env['_observables_dict']


def _finalize_sim_measures(exec_env: ExecEnv, sim_measures, g, save=True):
    """
    Common finalization logic for compute_g and compute_g_mp.
    
    Computes observable distances, optionally saves results, and returns sim_measures.
    
    Args:
        exec_env: Execution environment dictionary
        sim_measures: Dictionary containing simulated observable measures
        g: Global coupling parameter value
        save: Whether to save results to file (default: True)
        
    Returns:
        sim_measures with distances added
    """
    out_file = exec_env['out_file']
    observables = _get_observables_dict(exec_env)
    processed = exec_env['processed']

    sim_measures['g'] = g
    compute_observables_distances(sim_measures, processed, observables)
    
    if save:
        model_attribues = get_model_attributes(exec_env)
        for key, value in model_attribues.items():
            sim_measures[f'model_attr_{key}'] = value   
        hdf.savemat(out_file, sim_measures)
    
    return sim_measures


def _print_distances(exec_env: ExecEnv, sim_measures, g):
    """
    Print distance metrics for all observables.
    
    Args:
        exec_env: Execution environment dictionary
        sim_measures: Dictionary containing computed distances
        g: Global coupling parameter value
    """
    observables = _get_observables_dict(exec_env)
    for ds in observables:
        for dname in observables[ds].distance_measures:
            print(f" {ds} for g={g}: {sim_measures[f'dist_{ds}_{dname}']};", end='', flush=True)
    print()  # newline after all distances


def _try_load_previous(exec_env: ExecEnv, g):
    """
    Check if previous results exist and should be loaded.
    
    Args:
        exec_env: Execution environment dictionary
        g: Global coupling parameter value
        
    Returns:
        Tuple of (sim_measures, was_loaded) where sim_measures is the loaded data
        or None, and was_loaded indicates if data was successfully loaded.
    """
    out_file = exec_env['out_file']
    force_recomputations = exec_env.get('force_recomputations', False)
    
    if not force_recomputations and os.path.exists(out_file):
        print(f"Loading previous data for g={g}")
        return hdf.loadmat(out_file), True
    
    return None, False


def compute_g(exec_env: ExecEnv, g):
    sim_measures, was_loaded = _try_load_previous(exec_env, g)
    
    if not was_loaded:
        print(f"Starting computation for g={g}")
        set_seed(secrets.randbits(32))
        sim_measures = eval_one_param(exec_env, g)
        sim_measures = _finalize_sim_measures(exec_env, sim_measures, g)

    if 'verbose' in exec_env and exec_env['verbose']:
        _print_distances(exec_env, sim_measures, g)
    return sim_measures


def sweep_g_parallel(gs, base_exec_env: ExecEnv, out_file_pattern, nproc):
    """
    Sweep over G values in parallel using ProcessPoolExecutor with retry on failure.

    Submits in batches of nproc to limit peak memory from pickling exec_env per task.
    Only actually-failed G values are retried (not still-running ones).

    Args:
        gs: Array of G values to sweep
        base_exec_env: Base execution environment dict (out_file will be set per G)
        out_file_pattern: Format string for output file path, e.g. 'path/fitting_g_{}.mat'
        nproc: Number of parallel worker processes
    """
    remaining_gs = list(gs)

    while remaining_gs:
        batch = remaining_gs[:nproc]
        print(f'Submitting batch of {len(batch)}/{len(remaining_gs)} G values with {nproc} workers')

        with ProcessPoolExecutor(max_workers=nproc) as pool:
            future_to_g = {}
            for gf in batch:
                exec_env = ExecEnv(base_exec_env)
                exec_env['out_file'] = out_file_pattern.format(np.round(gf, decimals=3))
                future = pool.submit(compute_g, exec_env, gf)
                future_to_g[future] = gf

            failed_gs = []
            for future in as_completed(future_to_g):
                gf = future_to_g[future]
                try:
                    future.result()
                    print(f"Finished g={gf}")
                except Exception as exc:
                    print(f"Failed g={gf}: {exc}")
                    failed_gs.append(gf)

        # Next iteration: retry failed from this batch + remaining unbatched
        remaining_gs = failed_gs + remaining_gs[len(batch):]


def sweep_g_sequential(gs, base_exec_env: ExecEnv, out_file_pattern, nproc):
    """
    Sweep over G values sequentially, parallelizing subjects within each G using compute_g_mp.

    Args:
        gs: Array of G values to sweep
        base_exec_env: Base execution environment dict (out_file will be set per G)
        out_file_pattern: Format string for output file path, e.g. 'path/fitting_g_{}.mat'
        nproc: Number of parallel worker processes for subject simulation
    """
    for gf in gs:
        exec_env = ExecEnv(base_exec_env)
        exec_env['out_file'] = out_file_pattern.format(np.round(gf, decimals=3))
        _, sim_measures = compute_g_mp(exec_env, gf, nproc)
        if exec_env.get('verbose', True):
            _print_distances(exec_env, sim_measures, gf)


def process_empirical_subjects(bold_signals, observables: dict[str, ObservableConfig], verbose=True):
    """Process empirical BOLD signals (transposes from (n_rois, t_max) to (t_max, n_rois))."""
    return _process_signals(bold_signals, observables, verbose, transpose=True)


def executor_simulate_single_subject(n, exec_env: ExecEnv, g):
    try:
        set_seed(secrets.randbits(32))
        signal, bold = simulate_single_subject(exec_env, g)
        if exec_env.get('callback_simulate_single_subject', None) is not None:
            exec_env['callback_simulate_single_subject'](n, exec_env, signal, bold)
        return n, bold
    except Exception as e:
        raise RuntimeError(f"Error simulating subject {n}: {e}")


def executor_simulate_single_subject_raw(n, exec_env: ExecEnv, g):
    try:
        set_seed(secrets.randbits(32))
        return n, simulate(exec_env, g)
    except Exception as e:
        raise RuntimeError(f"Error simulating subject {n}: {e}")
        

def execute_multiprocessing_simulation(exec_env: ExecEnv, g, nproc, executor_func=None, result_key='simulated_data'):
    """
    Execute simulation for multiple subjects using multiprocessing.
    
    Args:
        exec_env: Execution environment dictionary
        g: Global coupling parameter
        nproc: Number of processes
        executor_func: Function to execute for each subject (default: executor_simulate_single_subject)
        result_key: Key name for the returned results dictionary (default: 'simulated_data')
        
    Returns:
        Tuple of (simulated_results, numerical_error)
    """
    if executor_func is None:
        executor_func = executor_simulate_single_subject
        
    out_file = exec_env['out_file']

    if os.path.exists(out_file):
        print(f"File {out_file} already exists, skipping...")
        return {}, False

    print(f'Computing executor for G={g}', flush=True)

    subjects = list(range(exec_env['num_subjects']))
    results = []
    print(f'Creating process pool with {nproc} workers')
    pending = subjects
    while len(pending) > 0:
        print(f"EXECUTOR --- START cycle for {len(pending)} subjects")
        with ProcessPoolExecutor(max_workers=nproc) as pool:
            futures = []
            future2subj = {}
            for n in pending:
                f = pool.submit(executor_func, n, exec_env, g)
                future2subj[f] = n
                futures.append(f)

            print(f"EXECUTOR --- WAITING for {len(futures)} futures to finish")

            pending = []
            for future in as_completed(futures):
                try:
                    n, result = future.result()
                    results.append((n, result))
                    print(f"EXECUTOR --- FINISHED subject {n}")
                except Exception as exc:
                    n = future2subj[future]
                    print(f"EXECUTOR --- FAIL subject {n}. Cause: {exc}. Restarting pool.")
                    pool.shutdown(wait=True, cancel_futures=True)
                    finished = [n for n, _ in results]
                    pending = [n for n in subjects if n not in finished]
                    break

    simulated_results = {}
    numerical_error = False
    for n, r in results:
        simulated_results[n] = r
        if np.any(np.isnan(simulated_results[n])) or np.any(np.isinf(simulated_results[n])):
            numerical_error = True
    
    return simulated_results, numerical_error


def execute_multiprocessing_simulation_bold(exec_env: ExecEnv, g, nproc):
    """
    Execute BOLD simulation for multiple subjects using multiprocessing.
    
    Args:
        exec_env: Execution environment dictionary
        g: Global coupling parameter
        nproc: Number of processes
        
    Returns:
        Tuple of (simulated_bolds, numerical_error)
    """
    return execute_multiprocessing_simulation(exec_env, g, nproc, 
                                            executor_simulate_single_subject, 
                                            'simulated_bolds')


def execute_multiprocessing_simulation_raw(exec_env: ExecEnv, g, nproc):
    """
    Execute raw signal simulation for multiple subjects using multiprocessing.
    
    Args:
        exec_env: Execution environment dictionary
        g: Global coupling parameter
        nproc: Number of processes
        
    Returns:
        Tuple of (simulated_signals, numerical_error)
    """
    return execute_multiprocessing_simulation(exec_env, g, nproc, 
                                            executor_simulate_single_subject_raw, 
                                            'simulated_signals')


def compute_g_mp(exec_env: ExecEnv, g, nproc):
    out_file = exec_env['out_file']
    
    # Check for previously computed results
    sim_measures, was_loaded = _try_load_previous(exec_env, g)
    if was_loaded:
        return {}, sim_measures
    
    print(f"Starting computation for g={g}")
    simulated_bolds, numerical_error = execute_multiprocessing_simulation_bold(exec_env, g, nproc)

    if numerical_error:
        sim_measures = {"error": "Nan or Inf in bold signals"}
        print(f"EXECUTOR --- NUMERICAL ERROR for {out_file}")
        hdf.savemat(out_file, sim_measures)
    else:
        sim_measures = process_bold_signals(simulated_bolds, _get_observables_dict(exec_env), verbose=exec_env.get('verbose', True))
        sim_measures = _finalize_sim_measures(exec_env, sim_measures, g)

    return simulated_bolds, sim_measures


def compute_observables_distances(sim_measures, processed, observables):
    for ds in observables:
        for dname, dist in observables[ds].distance_measures.items():
            sim_measures[f'dist_{ds}_{dname}'] = dist.distance(sim_measures[ds], processed[ds])


def _extract_model_attrs(mat_data):
    """
    Extract model attributes from a loaded mat file.
    
    Args:
        mat_data: Dictionary loaded from mat file
        
    Returns:
        Dictionary of model attributes {attr_name: attr_value}
    """
    attrs = {}
    for key in mat_data.keys():
        if key.startswith('model_attr_'):
            # Extract the attribute name without the prefix
            attr_name = key[len('model_attr_'):]
            value = mat_data[key]
            # Convert numpy arrays to hashable types for grouping
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    value = tuple(value.flatten())
            attrs[attr_name] = value
    return attrs


def _attrs_to_key(attrs):
    """
    Convert attributes dictionary to a hashable key for grouping.
    
    Args:
        attrs: Dictionary of attributes
        
    Returns:
        Tuple of sorted (key, value) pairs
    """
    return tuple(sorted(attrs.items()))


def _attrs_to_label(attrs):
    """
    Convert attributes dictionary to a human-readable label.
    
    Args:
        attrs: Dictionary of attributes
        
    Returns:
        String label for the attribute combination
    """
    if not attrs:
        return "default"
    parts = [f"{k}={v}" for k, v in sorted(attrs.items())]
    return ", ".join(parts)


def _format_metric_label(dist_key):
    """Format a dist_key like 'dist_FC_PS' into a readable label like 'FC (PS)'."""
    label = dist_key.replace('dist_', '')
    parts = label.split('_')
    if len(parts) >= 2:
        return f"{parts[0]} ({parts[1]})"
    return label


def _plot_single_group(out_file_path, data, group_label, group_suffix, show=True):
    """
    Generate a Deco2018 Figure 3A-style plot for a group of fitting results.

    All distance metrics are overlaid on a single plot with dual y-axes
    (left axis for the first metric, right axis for the second).

    Args:
        out_file_path: Path where the plot will be saved
        data: Dictionary of {dist_key: [(g, dist_value), ...]}
        group_label: Human-readable label for the group
        group_suffix: Safe filename suffix for the group
        show: Whether to display the plot

    Returns:
        Path to the saved plot file
    """
    import matplotlib.pyplot as plt

    # Colors matching Deco2018 Fig 3A: red for FC, green for FCD, then blue, orange...
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']

    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_keys = sorted(data.keys())

    global_max = 0.0
    for idx, key in enumerate(sorted_keys):
        values = data[key]
        color = colors[idx % len(colors)]
        label = _format_metric_label(key)

        # Sort by G value
        sorted_values = sorted(values, key=lambda x: x[0])
        g_values = np.array([v[0] for v in sorted_values])
        dist_values = np.array([v[1] for v in sorted_values])
        global_max = max(global_max, np.max(dist_values))

        ax.plot(g_values, dist_values, '-', color=color, linewidth=2.5, label=label)

        # Mark optimum: minimum for KS-type distances, maximum for PS (correlation)
        if 'PS' in key:
            opt_idx = np.argmax(dist_values)
        else:
            opt_idx = np.argmin(dist_values)
        ax.axvline(x=g_values[opt_idx], color=color, linestyle='--', alpha=0.4, linewidth=1)

    ax.set_ylim(0, global_max * 1.05)
    ax.set_xlabel('Global Coupling', fontsize=12)
    ax.set_ylabel('Fitting', fontsize=12)

    if group_label != "default":
        fig.suptitle(group_label, fontsize=14, fontweight='bold')

    ax.legend(loc='upper center', fontsize=10, framealpha=0.8)

    fig.tight_layout()

    # Save the figure
    if group_suffix:
        plot_file = os.path.join(out_file_path, f'fitting_distances_plot_{group_suffix}.png')
    else:
        plot_file = os.path.join(out_file_path, 'fitting_distances_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return plot_file


def plot_fitting_distances(out_file_path, file_pattern, show=True):
    """
    Scan fitting result files and generate plots with distance metrics vs global coupling (G).
    
    Files are grouped by their model_attr_* attributes, and a separate plot is generated
    for each unique combination of attribute values.
    
    Args:
        out_file_path: Path containing fitting_g*.mat files and where the plot will be saved
        file_pattern: Glob pattern for finding fitting files
        show: Whether to display the plots (default: True)
        
    Returns:
        List of paths to saved plot files, or None if no data to plot
    """
    # Scan for fitting files and collect distance data grouped by attributes
    file_list = glob.glob(os.path.join(out_file_path, file_pattern))
    
    # Dictionary: {attrs_key: {dist_key: [(g, value), ...]}}
    grouped_data = {}
    # Dictionary: {attrs_key: attrs_dict} for labels
    attrs_lookup = {}
    
    for file in file_list:
        print(f"Found file: {file}", flush=True)
        m = hdf.loadmat(file)
        if 'g' not in m:
            print(f"   Skipping file {file} due to lack of g data", flush=True)
            continue
        
        g = m['g']
        attrs = _extract_model_attrs(m)
        attrs_key = _attrs_to_key(attrs)
        
        if attrs_key not in grouped_data:
            grouped_data[attrs_key] = {}
            attrs_lookup[attrs_key] = attrs
        
        for key in m.keys():
            if key.startswith('dist_'):
                if key not in grouped_data[attrs_key]:
                    grouped_data[attrs_key][key] = []
                grouped_data[attrs_key][key].append((g, m[key]))
    
    if len(grouped_data) == 0:
        print("No distance data found to plot.")
        return None
    
    print(f"\nFound {len(grouped_data)} attribute group(s) to plot:")
    for attrs_key in grouped_data:
        label = _attrs_to_label(attrs_lookup[attrs_key])
        print(f"  - {label}")
    print()
    
    # Generate a plot for each group
    plot_files = []
    for idx, (attrs_key, data) in enumerate(sorted(grouped_data.items())):
        attrs = attrs_lookup[attrs_key]
        group_label = _attrs_to_label(attrs)
        
        # Create a safe filename suffix from attributes
        if attrs:
            suffix_parts = [f"{k}_{v}" for k, v in sorted(attrs.items())]
            group_suffix = "_".join(suffix_parts)
            # Make filename safe by replacing problematic characters
            group_suffix = group_suffix.replace('.', 'p').replace(' ', '_').replace(',', '_')
        else:
            group_suffix = "" if len(grouped_data) == 1 else f"group_{idx}"
        
        plot_file = _plot_single_group(out_file_path, data, group_label, group_suffix, show)
        plot_files.append(plot_file)
    
    return plot_files


def run(args):

    out_file_path = args.out_path
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)

    if args.plot_g:
        plot_fitting_distances(out_file_path, "fitting_g*.mat")
        exit(0)

    # Check for some required parameters when not plotting
    if args.obs_var is None:
        raise RuntimeError("Please provide the --obs-var parameter with the name of the observable variable to simulate")
    
    if args.observables is None or len(args.observables) == 0:
        raise RuntimeError("Please provide at least one observable using the --observables parameter")

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
    if args.sc_sigma > 0.0:
        sc_norm = lambda : sc_norm + np.random.normal(loc=0.0, scale=args.sc_sigma*np.max(sc_norm), size=sc_norm.shape)

    bold = True
    if args.model not in ModelFactory.list_available_models():
        raise ValueError(f"Model <{args.model}> not supported!")

    model = ModelFactory.create_model(args.model)
    if args.obs_var not in model.state_vars and args.obs_var not in model.observable_vars:
        raise ValueError(f"Observable variable <{args.obs_var}> not found in model!")
    
    bpf = BandPassFilter(tr=tr, k=args.bpf[0], flp=args.bpf[1], fhi=args.bpf[2]) if args.bpf is not None else None

    all_fMRI = {s: d for s, d in enumerate(timeseries)}

    # Process (or load) empirical data
    emp_filename = os.path.join(out_file_path, 'fNeuro_emp.mat')
    observables = create_observables_dict(args.observables, bpf)
    if not os.path.exists(emp_filename):
        processed = process_empirical_subjects(all_fMRI, observables)
        hdf.savemat(emp_filename, processed)
    else:
        processed = {observable_name: load_2d_matrix(emp_filename, index=observable_name) for observable_name in observables.keys()}

    out_file_name_pattern = os.path.join(out_file_path, 'fitting_g_{}.mat')

    n_subj = args.nsubj if args.nsubj is not None else n_frmis

    bold_model = BoldModelFactory.create_model(args.bold_model)

    if args.g is not None and not args.use_mp:
        # Single point execution for debugging purposes
        exec_env: ExecEnv = {
            'verbose': True,
            'model': args.model,
            'dt': dt,
            'weights': sc_norm,
            'processed': processed,
            'tr': tr,
            'observables': args.observables,
            'obs_var': args.obs_var,
            'bold': bold,
            'bold_model': bold_model,
            'out_file': out_file_name_pattern.format(np.round(args.g, decimals=3)),
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
            'force_recomputations': False,
        }
        compute_g(exec_env, args.g)

    elif args.g is not None and args.use_mp:
        exec_env: ExecEnv = {
            'verbose': True,
            'model': args.model,
            'dt': dt,
            'weights': sc_norm,
            'processed': processed,
            'tr': tr,
            'observables': args.observables,
            'obs_var': args.obs_var,
            'bold': bold,
            'bold_model': bold_model,
            'out_file': out_file_name_pattern.format(np.round(args.g, decimals=3)),
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
        }
        compute_g_mp(exec_env, args.g, args.nproc)

    elif args.g_range is not None:
        [g_Start, g_End, g_Step] = args.g_range
        gs = np.arange(g_Start, g_End + g_Step, g_Step)

        base_exec_env: ExecEnv = {
            'verbose': True,
            'model': args.model,
            'dt': dt,
            'weights': sc_norm,
            'processed': processed,
            'tr': tr,
            'observables': args.observables,
            'obs_var': args.obs_var,
            'bold': bold,
            'bold_model': bold_model,
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
        }
        sweep_g_parallel(gs, base_exec_env, out_file_name_pattern, args.nproc)

    elif args.param is not None:
        # Parameter exploration
        param_explore = parse_parameter_definitions(args.param)
        
        # Create base execution environment
        base_exec_env: ExecEnv = {
            'verbose': True,
            'model': args.model,
            'dt': dt,
            'weights': sc_norm,
            'processed': processed,
            'tr': tr,
            'observables': args.observables,
            'obs_var': args.obs_var,
            'bold': bold,
            'bold_model': bold_model,
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
            'force_recomputations': False,
            'scale_signal': args.scale_signal,
        }
        
        # Run parameter exploration
        run_parameter_exploration(param_explore, base_exec_env, out_file_path, args.nproc)

    else:
        RuntimeError("Neither --g, --g-range, nor --param has been defined")


def parse_parameter_definitions(param_args):
    """
    Parse parameter definitions from command line arguments.
    
    Args:
        param_args: List of parameter definition arguments
        
    Returns:
        Dictionary mapping parameter names to values/ranges
    """
    param_explore = {}
    
    for p in param_args:
        if len(p) < 3:
            raise RuntimeError(f"Parameter definition <{p}> is not valid. Lack of data")
        
        try:
            param_type = p[0]
            param_name = p[1]
            
            if param_type not in ['single', 'range', 'list']:
                raise RuntimeError(f"Parameter definition <{p}> is not valid. Unknown parameter type: {param_type}")
            
            if param_type == 'single':
                if len(p) != 3:
                    raise RuntimeError(f"Parameter definition <{p}> is not valid. Single parameters require a name, type, and value")
                try:
                    param_value = ast.literal_eval(p[2])
                except (ValueError, SyntaxError):
                    raise RuntimeError(f"Parameter definition <{p}> is not valid. Cannot parse single value: {p[2]}")
                param_explore[param_name] = param_value

            elif param_type == 'range':
                if len(p) != 5:
                    raise RuntimeError(f"Parameter definition <{p}> is not valid. Range parameters require a name, type, start, end and step")
                pv_start, pv_end, pv_step = map(float, p[2:])
                param_explore[param_name] = np.arange(pv_start, pv_end + pv_step, pv_step)
                
            else:  # list
                if len(p) < 3:
                    raise RuntimeError(f"Parameter definition <{p}> is not valid. List parameters require a name, type and at least one value")
                p_values = list(map(float, p[2:]))
                param_explore[param_name] = p_values
                
        except Exception as e:
            raise RuntimeError(f"Failed to parse parameter {param_name}: {e}")
    
    return param_explore


def generate_parameter_combinations(param_explore):
    """
    Generate all combinations of parameter values.
    
    Args:
        param_explore: Dictionary mapping parameter names to values/ranges
        
    Returns:
        List of parameter combinations as lists of (name, value) tuples
    """
    param_names = list(param_explore.keys())
    param_values = [
        param_explore[name] if isinstance(param_explore[name], (list, np.ndarray)) 
        else [param_explore[name]] 
        for name in param_names
    ]
    
    # Generate all combinations of parameter values
    param_combinations = list(itertools.product(*param_values))
    
    # Convert to list of lists of tuples (parameter name, value)
    all_param_sets = []
    for combination in param_combinations:
        param_set = [(param_names[i], combination[i]) for i in range(len(param_names))]
        all_param_sets.append(param_set)
    
    return all_param_sets


def process_model_parameters(param_set):
    """
    Process parameter set to separate model parameters from global coupling.
    
    Args:
        param_set: List of (name, value) tuples
        
    Returns:
        Tuple of (model_params_dict, g_value)
    """
    model_params_cli = [p for p in param_set if p[0] != 'g']
    model_params = []
    
    for p in model_params_cli:
        pname = p[0]
        if ',' not in pname:
            model_params.append(p)
        else:
            # Handle hyphenated parameters by splitting
            ps = p[0].split(',')
            for pp in ps:
                model_params.append((pp, p[1]))
    
    g_values = [p[1] for p in param_set if p[0] == 'g']
    g = g_values[0] if g_values else None
    
    return dict(model_params), g


def generate_parameter_filename(param_set, prefix='fitting'):
    """
    Generate a filename based on parameter values.
    
    Args:
        param_set: List of (name, value) tuples
        prefix: Filename prefix
        
    Returns:
        Generated filename
    """
    fname = prefix
    for p in param_set:
        x = p[1]
        if isinstance(x, (bool, int)):        
            fname += f"_{p[0]}_{p[1]}"
        elif isinstance(x, float):
            fname += f"_{p[0]}_{np.round(p[1], decimals=2)}"
    fname += '.mat'
    return fname


def run_parameter_exploration(param_explore, base_exec_env: ExecEnv, out_file_path, nproc, callback=None):
    """
    Run parameter exploration with all combinations.
    
    Args:
        param_explore: Dictionary of parameters to explore
        base_exec_env: Base execution environment
        out_file_path: Output directory path
        nproc: Number of processes
    """
    all_param_sets = generate_parameter_combinations(param_explore)
    
    print(f"Generated {len(all_param_sets)} parameter combinations")
    
    for param_set in all_param_sets:
        print(f"Combination: {param_set}")
        
        model_params, g = process_model_parameters(param_set)
        
        if g is None:
            print("Warning: No 'g' parameter found in parameter set, skipping...")
            continue
        
        fname = generate_parameter_filename(param_set)
        out_file = os.path.join(out_file_path, fname)
        
        if not base_exec_env['force_recomputations'] and os.path.exists(out_file):
            print(f"File {out_file} already exists, skipping...")
            continue

        # Create execution environment with model parameters
        exec_env = copy.deepcopy(base_exec_env)
        exec_env['model_attributes'] = model_params
        exec_env['out_file'] = out_file
        
        # Run the computation
        bolds, sim_measures = compute_g_mp(exec_env, g, nproc)

        if callback:
            callback(bolds, sim_measures, exec_env)

def gen_arg_parser():
    parser = argparse.ArgumentParser(description="Global coupling fitting script for NeuroNumba models.")
    parser.add_argument("--full-scan", action='store_true', default=False, help="Full scan all models/observables/measures")
    parser.add_argument("--verbose", action='store_true', default=False, help="Prints extra information during execution")
    parser.add_argument("--use-mp", action='store_true', default=False, help="Use multiprocessing if possible")
    parser.add_argument("--nproc", type=int, default=10, help="Number of parallel processes")
    parser.add_argument("--nsubj", type=int, help="Number of subjects for the simulations")
    parser.add_argument("--g", type=float, help="Single point execution for a global coupling value")
    parser.add_argument("--g-range", nargs=3, type=float, help="Parameter sweep range for G (start, end, step)")
    parser.add_argument("--bpf", nargs=3, type=float, help="Band pass filter to apply to BOLD signal (k, lp freq, hp freq)")
    parser.add_argument("--model", type=str, default='Deco2014', help="Model to use (Hopf, Deco2014, Montbrio, Zerlaut1O, Zerlaut2O)")
    parser.add_argument("--obs-var", type=str, help="Model variable to observe")
    parser.add_argument("--observables", nargs='+', type=str, help="Pairs (comma separated) of observables,distance to use (FC, phFCD, swFCD),(PS, KS)")
    parser.add_argument("--bold-model", type=str, default='Stephan2007', help="BOLD Model to use (Stephan2008, Stephan2007, Stephan2007Alt)")
    parser.add_argument("--out-path", type=str, required=True, help="Path to folder for output results")
    parser.add_argument("--tr", type=float, help="Time resolution of fMRI scanner (seconds)")
    parser.add_argument("--sc-scaling", type=float, default=0.2, help="Scaling factor for the SC matrix")
    parser.add_argument("--sc-sigma", type=float, default=0.0, help="Sigma scale value to generate noise for the SC matrix")
    parser.add_argument("--scale-signal", type=float, default=1.0, help="Scaling signal factor for unit conversion")
    parser.add_argument("--tmax", type=float, help="Override simulation time (seconds)")
    parser.add_argument("--fmri-path", type=str, help="Path to fMRI timeseries data")
    parser.add_argument("--plot-g", action='store_true', default=False, help="Plot G optimization results")
    parser.add_argument("--param", action='append', nargs="+", type=str, help="Parameter values to use in the model, e.g. --param single tau_e 10.0 --param range J_ee 5.0 15.0 1.0")

    return parser

if __name__ == '__main__':
    parser = gen_arg_parser()
    args = parser.parse_args()
    run(args)

