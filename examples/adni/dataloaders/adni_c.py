# =====================================================================================
# Methods to input AD data
# Subjects: CN 3, MCI 2, AD ? - in 2 longitudinal sessions each
# RoIs: 84 - TR = 3 - timepoints: 193
# Info for each subject: timeseries, SC
#
# The ADNI-3 dataset was graciously provided by Xenia Kobeleva
#
# =====================================================================================
import numpy as np
import os

from adni.dataloaders.base_dl import DataLoader
from neuronumba.basic.attr import Attr


class AdniC(DataLoader):
    classification = Attr(dependant=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._derivatives_folder = os.path.join(self.base_folder, 'bids/derivatives')
        self._fmri_folder = os.path.join(self._derivatives_folder, 'xcp_d')
        self.sc_folder = os.path.join(self._derivatives_folder, 'qsirecon')

        self._subj_info_file = os.path.join(self.base_folder, 'utils/df_adni_tau_amy_mri_longitud.csv')

        self.__loadAllData()

    def __loadAllData(self):
        # ----------- load general info
        self.info = np.loadtxt(self._subj_info_file, delimiter=',')
        classif = self.info[['PTID', 'TADAD_SES', 'DX']]
        classif = classif.reset_index()
        self.classification = {}
        for index, row in classif.iterrows():
            self.classification[(row['PTID'], 'ses-' + str(row['TADAD_SES']).zfill(2))] = row['DX']
        patients = list(dict.fromkeys(classif['PTID']))
        print(f"loaded: {patients}")


    def name(self):
        return 'ADNI_C'

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return 84

    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        # ----------- load fMRI data
        patient_fMRI_folder = os.path.join(self._fmri_folder, f'sub-{subjectID[0]}/{subjectID[1]}/func/sub-{subjectID[0]}_{subjectID[1]}_task-rest_run-1_space-MNI152NLin6Asym_res-01_desc-timeseries_desikan_killiany.tsv')
        timeseries = np.loadtxt(patient_fMRI_folder, delimiter='\t')
        print(f"loaded: {patient_fMRI_folder}")
        # ----------- load SC
        patient_SC_folder = os.path.join(self.sc_folder, f'sub-{subjectID[0]}/{subjectID[1]}/dwi/sub-{subjectID[0]}_desikan_sift_invnodevol_radius2_count_connectivity.csv')
        SC = np.loadtxt(patient_SC_folder, delimiter=',')
        SC = SC.drop('Unnamed: 0', axis=1)
        return {subjectID: {'timeseries': timeseries.T,
                            'SC': SC}}

    def get_parcellation(self):
        raise NotImplemented('We do not have this data yet!')
