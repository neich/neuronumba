# =====================================================================================
# Methods to input AD data
# Subjects: HC 17, MCI 9, AD 10 - RoIs: 379 - TR = 3 - timepoints: 197 (but 2 have 950-ish)
# Info for each subject: timeseries, ABeta, Tau, SC
#
# The ADNI-1 dataset was graciously provided by Leon Stefanovsi and Petra Ritter
#
# =====================================================================================
from sys import platform
import numpy as np
import pandas as pd
import os, csv

from dataloaders.base_dl import DataLoader
import dataloaders.parcellations.glasser379 as Glasser379

from neuronumba.basic.attr import Attr

def characterizeConnectivityMatrix(C):
    return np.max(C), np.min(C), np.average(C), np.std(C), np.max(np.sum(C, axis=0)), np.average(np.sum(C, axis=0))


# ===================== Normalize a SC matrix
normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum, as in Gus' codes
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std, maxNodeInput, avgNodeInput = characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print("  => impact=Avg*#:{}".format(avg*C.shape[0]), end='')
    print("  => maxNodeInputs:{}".format(maxNodeInput), end='')
    print("  => avgNodeInputs:{}".format(avgNodeInput))


# This is used to avoid (almost) "infinite" computations for some cases (i.e., subjects) that have fMRI
# data that is way longer than any other subject, causing almost impossible computations to perform,
# because they last several weeks (~4 to 6), which seems impossible to complete with modern Windows SO,
# which restarts the computer whenever it wants to perform supposedly "urgent" updates...
force_Tmax = True
BOLD_length = 197


# This method is to perform the timeSeries cutting when excessively long...
def cutTimeSeriesIfNeeded(timeseries, limit_forcedTmax=BOLD_length):
    if force_Tmax and timeseries.shape[1] > limit_forcedTmax:
        print(f"cutting lengthy timeseries: {timeseries.shape[1]} to {limit_forcedTmax}")
        timeseries = timeseries[:,0:limit_forcedTmax]
    return timeseries


# --------------------------------------------------
# Classify subject information into {HC, MCI, AD}
# --------------------------------------------------


dataSetLabels = ['HC', 'MCI', 'AD']


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class AdniA(DataLoader):
    use360 = Attr(default=False)
    correct_sc_matrix = Attr(default=True)
    normalize_burden = Attr(default=True)
    cut_timeseries = Attr(default=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_data()

    def _load_data(self):
        self._subjects = [os.path.basename(f.path) for f in os.scandir(os.path.join(self.base_folder, "connectomes")) if f.is_dir()]
        self.classification = self.check_classifications()
        HCSubjects = [s for s in self.classification if self.classification[s] == 'HC']
        ADSubjects = [s for s in self.classification if self.classification[s] == 'AD']
        MCISubjects = [s for s in self.classification if self.classification[s] == 'MCI']

    def name(self):
        return 'ADNI_A'

    def TR(self):
        return 3.

    def N(self):
        if self.use360:
            return 360
        return 379 # 360 cortical + 19 subcortical regions

    # get_fullGroup_data: convenience method to load all data for a given subject group
    def get_fullGroup_data(self, group):
        groupFMRI = self.load_fullCohort_fMRI(cohort=group)
        for s in groupFMRI:
            groupFMRI[s] = {'timeseries': cutTimeSeriesIfNeeded(groupFMRI[s])}
        return groupFMRI

    def get_AvgSC_ctrl(self, normalized=None):
        avgMatrix = self.computeAvgSC_HC_Matrix()
        if normalized is not None:
            return correctSC(avgMatrix)
        else:
            return avgMatrix

    def get_groupSubjects(self, group):
        return self.get_cohort_subjects(group)

    def get_groupLabels(self):
        return dataSetLabels

    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        # 1st load
        SCnorm, abeta_burden, tau_burden, timeseries = self.load_subject_data(subjectID)
        # 2nd cut
        timeseries = cutTimeSeriesIfNeeded(timeseries)[:self.N()]
        return {subjectID:
                    {'timeseries': timeseries,
                     'ABeta': abeta_burden,
                     'Tau': tau_burden,
                     'SC': SCnorm
                     }}

    def get_parcellation(self):
        return Glasser379.Glasser379()

    def load_all_HC_fMRI(self):
        return self.load_fullCohort_fMRI(cohort='HC')

    def load_fullCohort_fMRI(self, cohort='HC'):
        cohortSet = [subject for subject in self.classification.keys() if self.classification[subject] == cohort]
        all_fMRI = {}
        for subject in cohortSet:
            print(f"fMRI {cohort}: {subject}")
            fMRI_path = os.path.join(self.base_folder, "fMRI", subject, "MNINonLinear/Results/Restingstate")
            series = np.loadtxt(os.path.join(fMRI_path, subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt"), delimiter='\t')
            subcSeries = np.loadtxt(
                os.path.join(fMRI_path, subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt"), delimiter='\t')
            fullSeries = np.concatenate((series, subcSeries))
            all_fMRI[subject] = fullSeries
        return all_fMRI

    def get_cohort_subjects(self, cohort):
        return [s for s in self.classification if self.classification[s] == cohort]

    def check_classifications(self):
        # ============================================================================
        # This code is to check whether we have the information of the type of subject
        # They can be one of:
        # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
        # ============================================================================
        input_classification = csv.reader(open(os.path.join(self.base_folder, "subjects.csv"), 'r'))
        classification = dict((rows[0],rows[1]) for rows in input_classification)
        mistery = []
        for pos, subject in enumerate(self._subjects):
            if subject in classification:
                print('{}: Subject {} classified as {}'.format(pos, subject, classification[subject]))
            else:
                print('{}: Subject {} NOT classified'.format(pos, subject))
                mistery.append(subject)
        print("Misisng {} subjects:".format(len(mistery)), mistery)
        print()
        return classification

    def computeAvgSC_HC_Matrix(self):
        connectome_folder = os.path.join(self.base_folder, "connectomes")
        HC = [subject for subject in self.classification.keys() if self.classification[subject] == 'HC']
        print("SC + HC: {} (0)".format(HC[0]))
        sc_folder = os.path.join(connectome_folder, HC[0], "DWI_processing")
        SC = np.loadtxt(os.path.join(sc_folder, "connectome_weights.csv"))

        sumMatrix = SC
        for subject in HC[1:]:
            print("SC + HC: {}".format(subject))
            sc_folder = os.path.join(connectome_folder, subject, "DWI_processing")
            SC = np.loadtxt(os.path.join(sc_folder, "connectome_weights.csv"))
            sumMatrix += SC
        return sumMatrix / len(HC)  # but we normalize it afterwards, so we probably do not need this...

    def load_subject_data(self, subject):
        sc_folder = os.path.join(self.base_folder, 'connectomes', subject, "DWI_processing")
        SC = np.loadtxt(os.path.join(sc_folder, "connectome_weights.csv"))
        if self.correct_sc_matrix:
            SCnorm = self.correct_sc(SC)
        else:
            SCnorm = np.log(SC + 1)

        abeta_burden = self.load_burden(subject, "Amyloid")
        tau_burden = self.load_burden(subject, "Tau")

        fMRI_path = os.path.join(self.base_folder, "fMRI", subject, "MNINonLinear", "Results", "Restingstate")
        series = np.loadtxt(os.path.join(fMRI_path, subject, "_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt"))
        subcSeries = np.loadtxt(os.path.join(fMRI_path, subject, "_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt"))
        fullSeries = np.concatenate((series,subcSeries))

        return SCnorm, abeta_burden, tau_burden, fullSeries

    def load_burden(self, subject, modality):
        pet_path = os.path.join(self.base_folder, "PET_loads", subject, "PET_PVC_MG", modality)
        RH_pet = np.loadtxt(pet_path+"/"+"L."+modality+"_load_MSMAll.pscalar.txt")
        LH_pet = np.loadtxt(pet_path+"/"+"R."+modality+"_load_MSMAll.pscalar.txt")
        subcort_pet = np.loadtxt(pet_path+"/"+modality+"_load.subcortical.txt")[-19:]
        all_pet = np.concatenate((LH_pet,RH_pet,subcort_pet))
        if self.normalize_burden:
            normalizedPet = all_pet / np.max(all_pet)  # We need to normalize the individual burdens for the further optimization steps...
        else:
            normalizedPet = all_pet
        return normalizedPet


if __name__ == '__main__':
    DL = AdniA()
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('HC')
    s1 = DL.get_subjectData('002_S_0413')
    print('done! ;-)')
