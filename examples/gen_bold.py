import argparse
import os
from pathlib import Path
import time

import numpy as np

from global_coupling_fitting import IntegratorFactory, ModelFactory, generate_parameter_combinations, parse_parameter_definitions, process_model_parameters
from neuronumba.bold import BoldStephan2008
from neuronumba.bold.stephan_2007 import BoldStephan2007
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.simulator.models import Hopf, Deco2014
from neuronumba.simulator.models.montbrio import Montbrio
from neuronumba.simulator.monitors import RawSubSample, TemporalAverage
from neuronumba.simulator.simulator import Simulator
from neuronumba.tools import filterps, hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools.loader import load_2d_matrix

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Model to use (Hopf, Deco2014)", type=str, default='Deco2014')
    parser.add_argument("--obs-var", help="Observable variable to use", type=str, required=True)
    parser.add_argument("--tmax", help="Simulation time (seconds)", type=float, default=300.0)
    parser.add_argument("--bpf", help="Band-pass filter parameters: k, flp, fhi", type=float, nargs=3, default=[2, 0.005, 0.08])
    parser.add_argument("--tr", help="Temporal resolution (TR) for the BOLD signal (ms)", type=float, default=720.0)
    parser.add_argument("--g", help="Global scaling for SC matrix normalization", type=float, default=1.0)
    parser.add_argument("--out-path", help="Path to folder for output results", type=str, required=True)
    parser.add_argument("--sc-matrix", help="Path to SC matrix (Matlab, csv, npy, npz)", type=str, required=True)
    parser.add_argument("--sc-index", help="Index inside SC matrix file to use", type=str, required=False)
    parser.add_argument("--sc-scaling", help="Scaling factor for the SC matrix", type=float, default=0.2)
    parser.add_argument("--param", action='append', nargs="+", type=str, help="Parameter values to use in the model, e.g. --param single tau_e 10.0 --param range J_ee 5.0 15.0 1.0")

    args = parser.parse_args()

    # -------------------------- Load SC matrix
    mat0 = load_2d_matrix(args.sc_matrix, index=args.sc_index)
    weights = args.g * args.sc_scaling * mat0 / mat0.max()

    dt = 0.1
    tr = args.tr  # TR is already in milliseconds

    if args.model not in ModelFactory.list_available_models():
        raise ValueError(f"Model <{args.model}> not supported!")
    
    if args.param is None:
        raise ValueError("No parameters specified for exploration!")

    param_explore = parse_parameter_definitions(args.param)

    if len(param_explore) == 0:
        raise ValueError("No valid parameters found for exploration!")

    all_param_sets = generate_parameter_combinations(param_explore)

    if len(all_param_sets) > 1:
        raise ValueError("Only single-value parameter exploration is supported in this example!")

    param_set = all_param_sets[0]
    print(f"Combination: {param_set}")
        
    model_params, g = process_model_parameters(param_set)

    model = ModelFactory.create_model(args.model)
    if args.obs_var not in model.state_vars and args.obs_var not in model.observable_vars:
        raise ValueError(f"Observable variable <{args.obs_var}> not found in model!")
    
    model.set_attributes(model_params)
    obs_var = args.obs_var
    sampling_period = 1.0
    model.configure(weights=weights, g=g)

    bpf = BandPassFilter(tr=tr, k=args.bpf[0], flp=args.bpf[1], fhi=args.bpf[2]) if args.bpf is not None else None

    integrator = IntegratorFactory.create_integrator(args.model, dt)

    n_rois = weights.shape[0]
    sampling_period = 1.0
    lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
    speed = 1.0
    con = Connectivity(weights=weights, lengths=lengths, speed=speed)

    # coupling = CouplingLinearDense(weights=weights, delays=con.delays, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois)
    history = HistoryNoDelays()
    # mnt = TemporalAverage(period=1.0, dt=dt)
    # monitor = TemporalAverage(period=sampling_period, monitor_vars=model.get_var_info([obs_var]))
    monitor = TemporalAverage(period=sampling_period, monitor_vars=model.get_var_info([obs_var]))
    s = Simulator(connectivity=con, model=model, history=history, integrator=integrator, monitors=[monitor])
    start_time = time.perf_counter()
    # Convert to milliseconds
    s.run(0, args.tmax * 1000.0)
    t_sim = time.perf_counter() - start_time
    data = monitor.data(obs_var)
    # In Montbrio model, 1 r == 100.0 Hz
    # And discard the first 2s
    re = 100.0 * data[2000:, :]

    # Normalize input signal
    baseline = np.mean(re, axis=0)
    rn = (re - baseline[np.newaxis, :]) / baseline[np.newaxis, :]
    rn += baseline[np.newaxis, :]

    b_s2008 = BoldStephan2008(tr=tr).configure()
    bold_s2008 = b_s2008.compute_bold(rn, monitor.period)[10:, :]
    bold_filtered_s2008 = bpf.filter(bold_s2008) if bpf is not None else bold_s2008
    
    b_s2007 = BoldStephan2007(tr=tr).configure()
    bold_s2007 = b_s2007.compute_bold(rn, monitor.period)[10:, :]
    bold_filtered_s2007 = bpf.filter(bold_s2007) if bpf is not None else bold_s2007

    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
     # Save data to MAT file
    hdf.savemat(out_path / "gen.mat", {'bold_filtered_s2008': bold_filtered_s2008, 
                                       'bold_s2008': bold_s2008, 
                                       'bold_filtered_s2007': bold_filtered_s2007, 
                                       'bold_s2007': bold_s2007, 
                                       'raw': rn})