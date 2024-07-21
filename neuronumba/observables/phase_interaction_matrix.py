from numba import njit
import numpy as np
from scipy import signal
from scipy.signal import detrend


@njit
def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c


def phase_interaction_matrix(ts, discard_offset=10):
    phases = np.empty((ts.shape[1], ts.shape[0]))
    for n in range(ts.shape[1]):
        sd = detrend(ts[:, n], type='constant')
        Xanalytic = signal.hilbert(sd)
        phases[n, :] = np.angle(Xanalytic)
    return _phase_interaction_matrix(ts, phases, discard_offset)

@njit
def _phase_interaction_matrix(ts, phases, discard_offset=10):
    (t_max, n_rois) = ts.shape
    npattmax = t_max - (2 * discard_offset - 1)  # calculates the size of phfcd matrix
    # Data structures we are going to need...
    d_fc = np.zeros((n_rois, n_rois))
    # PhIntMatr = np.zeros((npattmax, int(N * (N - 1) / 2)))  # The int() is not needed, but... (see above)
    pim_matrix = np.zeros((npattmax, n_rois, n_rois))
    # syncdata = np.zeros(npattmax)

    T = np.arange(discard_offset, t_max - discard_offset + 1)
    for t in T:
        for i in range(n_rois):
            for j in range(i + 1):
                d_fc[i, j] = np.cos(adif(phases[i, t - 1], phases[j, t - 1]))
                d_fc[j, i] = d_fc[i, j]
        pim_matrix[t - discard_offset] = d_fc

    return pim_matrix
