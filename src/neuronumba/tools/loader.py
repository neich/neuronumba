import os

import numpy as np

import neuronumba.tools.hdf as hdf

def load_2d_matrix(filename, delimiter=None, index=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.csv':
        return np.loadtxt(filename, delimiter=delimiter)
    elif file_extension == '.tsv':
        if delimiter is None:
            delimiter = '\t'
        return np.loadtxt(filename, delimiter=delimiter)
    elif file_extension == '.mat':
        if index is None:
            raise RuntimeError("You have to provide an index for the file")
        return hdf.loadmat(filename)[index]
    elif file_extension == '.npy':
        return np.load(filename)
    elif file_extension == '.npz':
        return np.load(filename, allow_pickle=True)[index]
    else:
        raise RuntimeError("Unrecognized file extension")