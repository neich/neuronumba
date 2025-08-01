import numba as nb
import numpy as np
from scipy import signal


@nb.njit
def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c


def phase_interaction_matrix(ts, discard_offset=10):
    """
    Compute phase interaction matrix from time series data.
    
    :param ts: signal with shape (n_rois, n_time_samples)
    :param discard_offset: number of samples to discard from beginning and end
    :return: phase interaction matrix with shape (npattmax, n_rois, n_rois)
    """
    # Pre-compute phases using vectorized operations where possible
    phases = _compute_phases_hilbert(ts)
    return _phase_interaction_matrix(ts, phases, discard_offset)


def _compute_phases_hilbert(ts):
    """
    Compute phases using Hilbert transform.
    This function cannot be numba-optimized due to scipy.signal.hilbert dependency.
    """
    n_rois, n_samples = ts.shape
    phases = np.empty((n_rois, n_samples))
    
    # Vectorize the mean subtraction where possible
    ts_centered = ts - np.mean(ts, axis=1, keepdims=True)
    
    # Apply Hilbert transform row by row (cannot vectorize this part)
    for n in range(n_rois):
        Xanalytic = signal.hilbert(ts_centered[n, :])
        phases[n, :] = np.angle(Xanalytic)
    
    return phases


@nb.njit(nb.f8[:, :, :](nb.f8[:, :], nb.f8[:, :], nb.intc))
def _phase_interaction_matrix(ts, phases, discard_offset=10):
    n_rois, t_max = ts.shape
    npattmax = t_max - (2 * discard_offset - 1)  # calculates the size of phfcd matrix
    
    # Data structures we are going to need...
    pim_matrix = np.zeros((npattmax, n_rois, n_rois))

    # Process each time point
    for t_idx in range(npattmax):
        t = t_idx + discard_offset  # actual time index
        
        # Compute the phase interaction matrix for this time point
        for i in range(n_rois):
            for j in range(i, n_rois):  # Start from i instead of i+1 to include diagonal
                if i == j:
                    pim_matrix[t_idx, i, j] = 1.0  # Phase difference with itself is always 0, cos(0) = 1
                else:
                    phase_diff_cos = np.cos(adif(phases[i, t], phases[j, t]))
                    pim_matrix[t_idx, i, j] = phase_diff_cos
                    pim_matrix[t_idx, j, i] = phase_diff_cos  # Symmetric matrix

    return pim_matrix
