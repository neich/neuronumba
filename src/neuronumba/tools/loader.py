import os

import numpy as np

import neuronumba.tools.hdf as hdf

def load_2d_matrix(filename, delimeter=None, index=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    filename, file_extension = os.path.splitext(filename)
    if file_extension == '.csv':
        return np.loadtxt(filename, delimiter=delimeter)
    elif file_extension == '.mat':
        if index is None:
            raise RuntimeError("You have to provide an index inse the file")
        return hdf.loadmat(filename)[index]
    elif file_extension == '.npy':
        return np.load(filename)
    elif file_extension == '.npz':
        return np.load(filename, allow_pickle=True)[index]
    else:
        raise RuntimeError("Unrecognized file extension")