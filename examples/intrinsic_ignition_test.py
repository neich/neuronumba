import argparse
import os

import matplotlib.pyplot as plt

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import Intrinsic_Ignition
from neuronumba.tools.loader import load_2d_matrix


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model to use (LinearHopf, Hopf)", type=str, default='LinearHopf')
    args = parser.parse_args()
    return args


def load_data():
    # Let's load some fMRI data for the example
    subj_bolds_raw = {}
    fmri_path = "./Data_Raw/ebrains_popovych"
    for path in os.listdir(fmri_path):
        subject_path = os.path.join(fmri_path, path)
        if os.path.isdir(subject_path):
            fmri_file = os.path.join(subject_path, 'rfMRI_REST1_LR_BOLD.csv')
            if not os.path.isfile(fmri_file):
                raise FileNotFoundError(f"fMRI file <{fmri_file}> not found!")
            subj_bolds_raw[path] = load_2d_matrix(fmri_file) # Must be in (time, node) format

    tr = 720 # fMRI TR in milliseconds
    n_nodes = subj_bolds_raw[next(iter(subj_bolds_raw))].shape[1] # Number of nodes
    return subj_bolds_raw, tr, n_nodes


if __name__ == '__main__':
    args = parse_arguments()

    subj_bolds_raw, tr, n_nodes = load_data()

    # ========================================================================
    # We create the bandpass filter we will use for the signals
    bpf = BandPassFilter(
        k=2,
        flp=0.01,
        fhi=0.09,
        tr=tr,
        apply_detrend=True,
        apply_demean=True,
        remove_artifacts=False
    )

    # Let's apply the filter to all bold signals of the subjects
    filtered_bolds = {subj: bpf.filter(subj_bolds_raw[subj]) for subj in subj_bolds_raw.keys()}

    # ========================================================================
    # We will need to compute the empirical Intrinsic_Ignition
    # For each subject, let's compute its II
    for subj in filtered_bolds.keys():
        filtered_ts = filtered_bolds[subj]
        ii = Intrinsic_Ignition()
        ii.modality = Intrinsic_Ignition.EventBasedIntrinsicIgnition
        obs = ii.from_fmri(filtered_ts)
        igni_emp  = obs['mevokedinteg']
        meta_emp = obs['stdevokedinteg']

    # ========================================================================
    # And plot!!!
    plt.boxplot([igni_emp, meta_emp], labels=['Ignition', 'Meta'])
    plt.legend(loc='best')
    plt.show()
