# ==========================================================================
# Main AD GEC optimization
# By Gustavo Patow
# ==========================================================================
import argparse

from dataloaders.adni_c import AdniC
from neuronumba.fitting.gec.fitting_gec import calc_H_freq, calc_COV_emp, FitGEC
from neuronumba.simulator.models import Hopf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables import FC
# ----------- import the rest of the modules


# ==========================================================================
# ==========================================================================
# ==========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-folder", help="Base folder", type=str, required=True)

    args = parser.parse_args()  # for example, for a single test, use --we-range 1.0 1.1 1.0

    # -------------------------- Constants
    DL = AdniC(args.base_folder)
    TR = DL.TR()
    all_HC_data = DL.get_fullGroup_data('CN')
    all_HC_fMRI = {s: all_HC_data[s]['timeseries'] for s in all_HC_data}
    t_max = max([all_HC_fMRI[s].shape[1] for s in all_HC_fMRI])
    avg_SC = DL.get_AvgSC_ctrl(ctrl_label='CN', normalized='maxSC')
    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=TR, apply_detrend=True, apply_demean=True)
    h_freq = calc_H_freq(all_HC_fMRI, DL.N(), t_max, TR, bpf)
    hopf = Hopf(omega=h_freq, a=-0.02)

    fc = FC()

    # -------------------------- For each subject, compute its GEC
    for subj in DL.get_classification():
        subjData = DL.get_subjectData(subj)
        timeseries = subjData[subj]['timeseries']

        # COV_emp is the timelagged covariance of the empirical data
        COV_emp = calc_COV_emp(timeseries)
        FC_emp  = fc.from_fmri(timeseries)['FC']



        GEC = FitGEC().fitGEC(FC_emp, COV_emp, avg_SC, hopf, TR)

        # don't forget to save it :)
        # np.save(....., GEC)

    print('Done')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF