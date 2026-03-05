import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from global_coupling_fitting import ModelFactory, IntegratorFactory
from neuronumba.bold import BoldStephan2007Alt
from neuronumba.simulator.models import Deco2014
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools.loader import load_2d_matrix

in_file_path = "./Data_Raw"
out_file_path = "./Data_Produced"

# Register Deco2014 model and its integrator configuration in the factories
ModelFactory.add_model('Deco2014', lambda: Deco2014())
IntegratorFactory.add_integrator_config('Deco2014', lambda dt: EulerStochastic(dt=dt, sigmas=np.r_[2e-4, 2e-4]))


def load_connectivity():
    C = load_2d_matrix(os.path.join(in_file_path, 'Human_66.mat'), index='C')
    C = C / np.max(C) * 0.2
    return C


def run_bifurcation():
    C = load_connectivity()

    dt = 0.1
    sampling_period = 1.0
    t_max_neuronal = 10000
    t_warmup = 1000

    gs = np.arange(0.0, 6.05, 0.2)
    max_re_fic = np.zeros(len(gs))
    max_re_nofic = np.zeros(len(gs))

    for i, g in enumerate(gs):
        print(f'[{i+1}/{len(gs)}] g={g:.2f}')

        # With FIC
        model = ModelFactory.create_model('Deco2014').set_attributes({'g': g, 'auto_fic': True})
        integrator = IntegratorFactory.create_integrator('Deco2014', dt)
        signal = simulate_nodelay(model, integrator, C, 're',
                                  sampling_period=sampling_period,
                                  t_max_neuronal=t_max_neuronal,
                                  t_warmup=t_warmup)
        max_re_fic[i] = np.max(signal)

        # Without FIC
        model = ModelFactory.create_model('Deco2014').set_attributes({'g': g, 'auto_fic': False})
        integrator = IntegratorFactory.create_integrator('Deco2014', dt)
        signal = simulate_nodelay(model, integrator, C, 're',
                                  sampling_period=sampling_period,
                                  t_max_neuronal=t_max_neuronal,
                                  t_warmup=t_warmup)
        max_re_nofic[i] = np.max(signal)

        print(f'  max re: FIC={max_re_fic[i]:.2f} Hz, no FIC={max_re_nofic[i]:.2f} Hz')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gs, max_re_fic, 'o-', label='With FIC')
    ax.plot(gs, max_re_nofic, 's-', label='Without FIC')
    ax.set_xlabel('G (global coupling)')
    ax.set_ylabel('Max firing rate re (Hz)')
    ax.set_title('Deco2014 — Max firing rate vs G')
    ax.legend()
    plt.tight_layout()
    plt.show()


def _simulate_subject_fc(C, g, use_fic, dt, sampling_period, t_max_neuronal, t_warmup):
    """Worker function for a single subject: simulate -> BOLD -> filter -> FC."""
    model = ModelFactory.create_model('Deco2014').set_attributes(
        {'g': g, 'auto_fic': use_fic})
    integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2])
    signal = simulate_nodelay(model, integrator, C, 're',
                              sampling_period=sampling_period,
                              t_max_neuronal=t_max_neuronal,
                              t_warmup=t_warmup)

    bold_model = BoldStephan2007Alt()
    bds = bold_model.compute_bold(signal, sampling_period)
    del signal

    if np.isnan(bds).any():
        return None

    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2000.0)
    bds_filt = bpf.filter(bds)
    return np.corrcoef(bds_filt, rowvar=False)


def run_figure3(n_subjects=10, max_workers=None):
    """
    Reproduce Figure 3 from Deco et al. 2014.

    Sweeps global coupling G and plots the Pearson correlation between
    simulated and empirical FC, comparing FIC vs no-FIC conditions.
    For each G, multiple trials are simulated in parallel, BOLD signals
    are generated via the Balloon-Windkessel model, band-pass filtered,
    and averaged into a single FC matrix that is then compared to empirical FC.

    Results for each G are saved to Data_Produced/figure3_g_{g}.mat so that
    re-running skips already computed values.
    """
    C = load_connectivity()
    FC_emp = load_2d_matrix(os.path.join(in_file_path, 'Human_66.mat'), index='FC_emp')

    dt = 0.1
    sampling_period = 1.0  # ms
    t_max_neuronal = 50000  # ms
    t_warmup = 0

    os.makedirs(out_file_path, exist_ok=True)

    gs = np.arange(0.0, 6.05, 0.5)

    n_rois = C.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    fc_emp_vec = FC_emp[triu_idx]

    fc_corr_fic = np.zeros(len(gs))
    fc_corr_nofic = np.zeros(len(gs))

    for i, g in enumerate(gs):
        out_file = os.path.join(out_file_path, f'figure3_g_{g:.2f}.mat')

        if os.path.exists(out_file):
            saved = hdf.loadmat(out_file)
            fc_corr_fic[i] = float(saved['fc_corr_fic'])
            fc_corr_nofic[i] = float(saved['fc_corr_nofic'])
            print(f'[{i+1}/{len(gs)}] g={g:.2f}  loaded from cache  '
                  f'FC corr: FIC={fc_corr_fic[i]:.4f}, noFIC={fc_corr_nofic[i]:.4f}')
            continue

        save_data = {}
        for use_fic, result_arr, label in [(True, fc_corr_fic, 'FIC'),
                                           (False, fc_corr_nofic, 'noFIC')]:
            fc_accum = np.zeros((n_rois, n_rois))
            valid = 0

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_simulate_subject_fc, C, g, use_fic,
                                dt, sampling_period, t_max_neuronal, t_warmup): s
                    for s in range(n_subjects)
                }
                for future in as_completed(futures):
                    s = futures[future]
                    try:
                        fc_sim = future.result()
                    except Exception as exc:
                        print(f'  [{label}] subject {s+1}: failed ({exc})')
                        continue
                    if fc_sim is None:
                        print(f'  [{label}] subject {s+1}: NaN in BOLD, skipping')
                        continue
                    print(f'  [{label}] subject {s+1}: done')
                    fc_accum += fc_sim
                    valid += 1

            if valid > 0:
                fc_avg = fc_accum / valid
                fc_sim_vec = fc_avg[triu_idx]
                result_arr[i] = np.corrcoef(fc_sim_vec, fc_emp_vec)[0, 1]
                save_data[f'fc_{label}'] = fc_avg

        save_data['fc_corr_fic'] = fc_corr_fic[i]
        save_data['fc_corr_nofic'] = fc_corr_nofic[i]
        hdf.savemat(out_file, save_data)
        print(f'[{i+1}/{len(gs)}] g={g:.2f}  '
              f'FC corr: FIC={fc_corr_fic[i]:.4f}, noFIC={fc_corr_nofic[i]:.4f}')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gs, fc_corr_fic, 'o-', label='With FIC')
    ax.plot(gs, fc_corr_nofic, 's-', label='Without FIC')
    ax.set_xlabel('G (global coupling)')
    ax.set_ylabel('Correlation(FC sim, FC emp)')
    ax.set_title('Deco2014 — Figure 3: FC fitting vs G')
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_simulation(g):
    C = load_connectivity()

    dt = 0.1
    sampling_period = 1.0
    t_max_neuronal = 10000
    t_warmup = 1000

    model = ModelFactory.create_model('Deco2014').set_attributes({'g': g, 'auto_fic': True})
    integrator = IntegratorFactory.create_integrator('Deco2014', dt)

    signal = simulate_nodelay(model, integrator, C, 're',
                              sampling_period=sampling_period,
                              t_max_neuronal=t_max_neuronal,
                              t_warmup=t_warmup)

    print(f'Simulation completed: g={g}, shape={signal.shape} (timepoints x regions)')

    n_regions = signal.shape[1]
    subsample_step = int(10.0 / sampling_period)  # 1/100 s = 10 ms
    signal_sub = signal[::subsample_step, :]
    time = np.arange(signal_sub.shape[0]) * sampling_period * subsample_step
    fig, ax = plt.subplots(figsize=(14, 8))
    for r in range(n_regions):
        ax.plot(time, signal_sub[:, r], linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing rate (re)')
    ax.set_title(f'Deco2014 simulation — G={g}, {n_regions} regions')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deco2014 model workflows')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('bifurcation', help='Sweep G and plot max firing rate with/without FIC')

    sim_parser = subparsers.add_parser('simulate', help='Run a single simulation for a given G value')
    sim_parser.add_argument('g', type=float, help='Global coupling parameter value')

    fig3_parser = subparsers.add_parser('figure3',
                                        help='Reproduce Figure 3: FC correlation vs G with/without FIC')
    fig3_parser.add_argument('--subjects', type=int, default=10,
                             help='Number of simulated subjects per G value (default: 10)')
    fig3_parser.add_argument('--workers', type=int, default=None,
                             help='Max parallel workers (default: number of CPUs)')

    args = parser.parse_args()

    if args.command == 'bifurcation':
        run_bifurcation()
    elif args.command == 'simulate':
        run_simulation(args.g)
    elif args.command == 'figure3':
        run_figure3(n_subjects=args.subjects, max_workers=args.workers)
