import warnings

import numpy as np
import numba as nb
#import cupy as cp

from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.basic.attr import Attr
from neuronumba.observables.phase_interaction_matrix import phase_interaction_matrix

def calc_length(start, end, step):
    # This fails for a negative step e.g., range(10, 0, -1).
    # From https://stackoverflow.com/questions/31839032/python-how-to-calculate-the-length-of-a-range-without-creating-the-range
    return (end - start - 1) // step + 1


# @jit(nopython=True)
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix
    corr_mat = np.corrcoef(x.flatten(), y.flatten())
    # Return entry [0,1]
    return corr_mat[0,1]


class SwFCD(ObservableFMRI):

    window_step = Attr(default=3)
    window_size = Attr(default=30)
    discard_offset = Attr(default=10, required=False)

    def _compute_from_fmri(self, signal):  # Compute the FCD of an input BOLD signal
        # Use the transposed array for performance
        s = signal.T
        n_rois, t_max = s.shape
        last_window = t_max - self.window_size  # 190 = 220 - 30
        N_windows = calc_length(0, last_window, self.window_step)  # N_windows = len(np.arange(0, last_window, self.window_step))

        result = None
        if not np.isnan(s).any():  # No problems, go ahead!!!
            Isubdiag = np.tril_indices(n_rois, k=-1)  # Indices of triangular lower part of matrix

            # For each pair of sliding windows calculate the FC at t and t2 and
            # compute the correlation between the two.
            cotsampling = np.zeros((int(N_windows * (N_windows - 1) / 2)))
            kk = 0
            ii2 = 0
            for t in range(0, last_window, self.window_step):
                jj2 = 0
                sfilt = (s[:, t:t + self.window_size + 1]).T  # Extracts a (sliding) window between t and t+self.window_size (included)
                cc = np.corrcoef(sfilt, rowvar=False)  # Pearson correlation coefficients
                for t2 in range(0, last_window, self.window_step):
                    sfilt2 = (s[:, t2:t2 + self.window_size + 1]).T  # Extracts a (sliding) window between t2 and t2+self.window_size (included)
                    cc2 = np.corrcoef(sfilt2, rowvar=False)  # Pearson correlation coefficients
                    ca = pearson_r(cc[Isubdiag], cc2[Isubdiag])  # Correlation between both FC
                    if jj2 > ii2:  # Only keep the upper triangular part
                        cotsampling[kk] = ca
                        kk = kk + 1
                    jj2 = jj2 + 1
                ii2 = ii2 + 1

            result = cotsampling
        else:
            result = np.nan

        return {'swFCD': result}

    # ==================================================================
    # buildFullMatrix: given the output of from_fMRI, this function
    # returns the full matrix. Not needed, except for plotting and such...
    # ==================================================================
    def build_full_matrix(self, FCD_data):
        LL = FCD_data.shape[0]
        # T is size of the matrix given the length of the lower/upper triangular part (displaced by 1)
        T = int((1. + np.sqrt(1. + 8. * LL)) / 2.)
        fcd_mat = np.zeros((T, T))
        fcd_mat[np.triu_indices(T, k=1)] = FCD_data
        fcd_mat += fcd_mat.T
        return fcd_mat