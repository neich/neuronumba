import h5py
import numpy as np
import scipy.io as sio


def loadmat(filename):
    try:
        f = h5py.File(filename, 'r')
        r = {}
        for k, ds in f.items():
            r[k] = np.array(ds[:])
        return r
    except:
        return sio.loadmat(filename)


def savemat(filename, mat):
    with h5py.File(filename, 'w') as f:
        for k, v in mat.items():
            f.create_dataset(k, data=v)
