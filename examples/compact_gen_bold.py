import argparse
import os
import time
import math

import numpy as np
from matplotlib import pyplot as plt

# If need to debug numba code, uncomment this
# from numba import config
# config.DISABLE_JIT = True

from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.compact_bold_simulator import CompactHopfSimulator, CompactDeco2014Simulator, CompactMontbrioSimulator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Model to use (Hopf, Deco2014, Montbrio)", type=str, default='Hopf')
    parser.add_argument("--tmax", help="Simulation time (milliseconds)", type=float, default=10000.0)
    parser.add_argument("--tr", help="Temporal resolution (TR) for the BOLD signal (milliseconds)", type=float, default=2000.0)
    parser.add_argument(
        "--dt",
        help=(
            "Simulation delta-time (milliseconds). "
            "Note: Hopf will be in the order of 10-100 (as it internally integrated in seconds), "
            "while other models (eg. Deco2014 or Montbrio) "
            "will be in the order of 0.1 (as they are internally integrated in milliseconds)"
        ),
        type=float,
        default=100
    )
    parser.add_argument("--g", help="Global scaling for SC matrix normalization", type=float, default=1.0)

    args = parser.parse_args()

    # The number of nodes
    n_rois = 70

    # We generate a Mock-up structural connectivity (SC) matrix for the purpose of the example. In a real-world scenario
    # you should use the real one. 
    sc_norm = np.random.uniform(0.05, 0.2, size=(n_rois, n_rois))
    np.fill_diagonal(sc_norm, 0.0)

    compact_simulator = None

    if args.model == 'Hopf':
        # For the hopf model, we need the intrinsic frequencies for each node. 
        # Here we do a mock-up version, but in real-world scenario you should perform
        # something similar to the fitting_gep example.
        omega = np.random.uniform(0.04, 0.07, size=n_rois)

        compact_simulator = CompactHopfSimulator(
            weights = sc_norm,
            use_temporal_avg_monitor = False, # If true, then it will average across sub-samples
            a = -0.02,
            omega = omega,
            g = args.g,
            sigma = 1e-03,
            tr = args.tr,
            dt = args.dt
        )
    elif args.model == 'Deco2014':
        compact_simulator = CompactDeco2014Simulator(
            weights = sc_norm,
            use_temporal_avg_monitor = False,
            g = args.g,
            sigma = 1e-03,
            tr = args.tr,
            dt = args.dt
        )
    elif args.model == 'Montbrio':
        compact_simulator = CompactMontbrioSimulator(
            weights = sc_norm,
            use_temporal_avg_monitor = False,
            g = args.g,
            sigma = 1e-03,
            tr = args.tr,
            dt = args.dt
        )
    else:
        raise RuntimeError(f"Unknown model <{args.model}>")

    simulated_bold = compact_simulator.generate_bold(
        warmup_samples = 100, # This samples will be discarded
        simulated_samples = math.ceil(args.tmax / args.dt)  # Number of useful samples to generate, this will be the size of the generated bold 
    )

    fig, axs = plt.subplots(1)
    fig.suptitle(f'Result for model {args.model} (g={args.g})')
    axs.plot(np.arange(simulated_bold.shape[0]), simulated_bold)
    plt.show()

