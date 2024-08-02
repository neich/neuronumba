import h5py
import numpy as np
import scipy.io as sio


def loadmat(filename):
    try:
        f = h5py.File(filename, 'r')
        r = {}
        for k, ds in f.items():
            if len(ds.shape) == 0:
                r[k] = ds[()]
            else:
                r[k] = np.array(ds[:])
        return r
    except Exception as e:
        return sio.loadmat(filename)


def savemat(filename, data):
    f = h5py.File(filename, 'w')
    for k, v in data.items():
        f.create_dataset(k, data=v)
    f.close()

