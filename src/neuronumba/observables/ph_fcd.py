import numpy as np
import numba as nb

from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.basic.attr import Attr
from neuronumba.observables.phase_interaction_matrix import phase_interaction_matrix


class PhFCD(ObservableFMRI):

    discard_offset = Attr(default=10, required=False)

    def _compute_from_fmri(self, bold_signal):  # Compute the FCD of an input BOLD signal
        # Use the transposed array for performance
        s = bold_signal.T
        n_rois, t_max = s.shape
        pim_matrix = phase_interaction_matrix(s, self.discard_offset)  # Compute the Phase-Interaction Matrix
        return PhFCD_from_fmri(n_rois, t_max, self.discard_offset, pim_matrix)

    # ==================================================================
    # buildFullMatrix: given the output of from_fMRI, this function
    # returns the full matrix. Not needed, except for plotting and such...
    # ==================================================================
    def buildFullMatrix(self, FCD_data):
        """
        Build full matrix from triangular FCD data.
        Optimized with numba for better performance.
        """
        return _build_full_matrix_numba(FCD_data)


@nb.njit
def _build_full_matrix_numba(FCD_data):
    """
    Numba-optimized function to build full matrix from triangular data.
    """
    LL = FCD_data.shape[0]
    # T is size of the matrix given the length of the lower/upper triangular part (displaced by 1)
    T = int((1.0 + np.sqrt(1.0 + 8.0 * LL)) / 2.0)
    fcd_mat = np.zeros((T, T))
    
    # Fill upper triangular part
    idx = 0
    for i in range(T):
        for j in range(i + 1, T):
            fcd_mat[i, j] = FCD_data[idx]
            fcd_mat[j, i] = FCD_data[idx]  # Make symmetric
            idx += 1
    
    return fcd_mat


def PhFCD_from_fmri(n_rois, t_max, discard_offset, pim_matrix):
    """
    Compute Phase FCD from phase interaction matrix.
    Optimized version that uses numba for the entire computation.
    """
    npattmax = t_max - (2 * discard_offset - 1)
    
    # Pre-compute triangular indices once
    tril_indices = _get_lower_triangular_indices(n_rois)
    
    # Use optimized numba function for the entire computation
    phfcd = _compute_phfcd_optimized(pim_matrix, npattmax, n_rois, tril_indices)
    
    return {'phFCD': phfcd}


@nb.njit
def _get_lower_triangular_indices(n_rois):
    """
    Get lower triangular indices for matrix extraction.
    Returns flattened indices for numba compatibility.
    """
    indices = np.empty((int(n_rois * (n_rois - 1) / 2), 2), dtype=np.int32)
    idx = 0
    for i in range(n_rois):
        for j in range(i):
            indices[idx, 0] = j  # row
            indices[idx, 1] = i  # col
            idx += 1
    return indices


@nb.njit
def _compute_phfcd_optimized(pim_matrix, npattmax, n_rois, tril_indices):
    """
    Optimized computation of Phase FCD using numba.
    """
    n_pairs = int(n_rois * (n_rois - 1) / 2)
    size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)
    phfcd = np.zeros(size_kk3)
    
    # Extract upper triangular part for all time points at once
    pim_up_tri = np.zeros((npattmax, n_pairs))
    for t in range(npattmax):
        for idx in range(n_pairs):
            i, j = tril_indices[idx, 0], tril_indices[idx, 1]
            pim_up_tri[t, idx] = pim_matrix[t, i, j]
    
    # Pre-compute all sums to avoid repeated computation
    window_sums = np.zeros((npattmax - 2, n_pairs))
    window_norms = np.zeros(npattmax - 2)
    
    for t in range(npattmax - 2):
        for idx in range(n_pairs):
            window_sums[t, idx] = pim_up_tri[t, idx] + pim_up_tri[t + 1, idx] + pim_up_tri[t + 2, idx]
        
        # Compute norm
        norm_sq = 0.0
        for idx in range(n_pairs):
            norm_sq += window_sums[t, idx] * window_sums[t, idx]
        window_norms[t] = np.sqrt(norm_sq)
    
    # Compute Phase FCD with pre-computed values
    kk3 = 0
    for t in range(npattmax - 2):
        for t2 in range(t + 1, npattmax - 2):
            # Compute dot product
            dot_product = 0.0
            for idx in range(n_pairs):
                dot_product += window_sums[t, idx] * window_sums[t2, idx]
            
            # Avoid division by zero
            if window_norms[t] > 0.0 and window_norms[t2] > 0.0:
                phfcd[kk3] = dot_product / (window_norms[t] * window_norms[t2])
            else:
                phfcd[kk3] = 0.0
            
            kk3 += 1
    
    return phfcd