# =====================================================================================
# Methods to input Glasser379 Parcellation data
#
# =====================================================================================
import numpy as np
import pandas as pd

from dataloaders.parcellations.parcellation import Parcellation
from neuronumba.tools import hdf


class Glasser379(Parcellation):
    def __init__(self, path):
        self.parcellation_folder = path

    def get_coords(self):
        # ----------------- coordinates, but only for the 360 version...
        cog = np.loadtxt(self.parcellation_folder + 'Glasser360_coords.txt')
        return cog

    def get_region_labels(self):
        # ----------------- node labels
        with open(self.parcellation_folder + 'glasser379NodeNames.txt', 'r') as file:
            node_names = [line.strip() for line in file]
        # ----------------- node long labels
        columnames = ['id', 'reg name']
        df = pd.read_csv(self.parcellation_folder + 'Anatomical-Labels.csv', names=columnames, header=None)
        nlist = df[columnames[1]].tolist()
        fullnames = nlist + nlist + node_names[360:378] + ['Brainstem']
        return fullnames

    def get_region_short_labels(self):
        # ----------------- node labels
        with open(self.parcellation_folder + 'glasser379NodeNames.txt', 'r') as file:
            node_names = [line.strip() for line in file]
        return node_names

    def get_cortices(self):
        # ----------------- Cortices
        df = pd.read_csv(self.parcellation_folder + 'HCP-MMP1_UniqueRegionList.csv')
        clist = df['cortex'][0:180].tolist()
        cortex = clist + clist + ['Subcortical'] * 18 + ['Brainstem']
        return cortex

    def get_RSN(self, useLR=False):
        raise NotImplemented('Unfinished implementation!')
        indicesFileParcellationRSN = f'../../Data_Produced/Parcellations/Glasser360RSN_{"14" if useLR else "7"}_indices.csv'

