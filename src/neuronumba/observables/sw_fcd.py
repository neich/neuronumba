import warnings

import numpy as np
import numba as nb
#import cupy as cp

from neuronumba.observables.base_observable import ObservableFMRI
from neuronumba.basic.attr import Attr
from neuronumba.observables.phase_interaction_matrix import phase_interaction_matrix

@nb.njit
def calc_length_numba(start, end, step):
    """Numba-optimized version of calc_length."""
    return (end - start - 1) // step + 1


@nb.njit
def pearson_r_numba(x, y):
    """
    Numba-optimized Pearson correlation coefficient computation.
    """
    n = x.shape[0]
    
    # Compute means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Compute correlation
    num = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        num += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy
    
    # Avoid division by zero
    if sum_sq_x == 0.0 or sum_sq_y == 0.0:
        return 0.0
    
    return num / np.sqrt(sum_sq_x * sum_sq_y)


@nb.njit
def corrcoef_numba(data):
    """
    Numba-optimized correlation coefficient matrix computation.
    data: shape (n_samples, n_features)
    returns: correlation matrix (n_features, n_features)
    """
    n_samples, n_features = data.shape
    corr_matrix = np.zeros((n_features, n_features))
    
    # Compute means for each feature
    means = np.zeros(n_features)
    for j in range(n_features):
        sum_val = 0.0
        for i in range(n_samples):
            sum_val += data[i, j]
        means[j] = sum_val / n_samples
    
    # Pre-compute centered data for efficiency
    centered_data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            centered_data[i, j] = data[i, j] - means[j]
    
    # Compute correlation matrix
    for i in range(n_features):
        corr_matrix[i, i] = 1.0  # Diagonal elements
        
        for j in range(i + 1, n_features):
            # Compute correlation between features i and j
            num = 0.0
            sum_sq_i = 0.0
            sum_sq_j = 0.0
            
            for k in range(n_samples):
                di = centered_data[k, i]
                dj = centered_data[k, j]
                num += di * dj
                sum_sq_i += di * di
                sum_sq_j += dj * dj
            
            if sum_sq_i == 0.0 or sum_sq_j == 0.0:
                correlation = 0.0
            else:
                correlation = num / np.sqrt(sum_sq_i * sum_sq_j)
            
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation
    
    return corr_matrix


@nb.njit
def extract_lower_triangular_numba(matrix, n_rois):
    """
    Extract lower triangular part of a symmetric matrix.
    """
    n_elements = int(n_rois * (n_rois - 1) / 2)
    result = np.zeros(n_elements)
    idx = 0
    
    for i in range(n_rois):
        for j in range(i):
            result[idx] = matrix[i, j]
            idx += 1
    
    return result


class SwFCD(ObservableFMRI):

    window_step = Attr(default=3)
    window_size = Attr(default=30)
    discard_offset = Attr(default=10, required=False)

    def _compute_from_fmri(self, signal):  # Compute the FCD of an input BOLD signal
        # Use the transposed array for performance
        s = signal.T
        n_rois, t_max = s.shape
        
        # Check for NaN values
        if np.isnan(s).any():
            return {'swFCD': np.nan}
        
        # Use optimized numba computation
        result = _compute_swfcd_optimized(s, n_rois, t_max, self.window_size, self.window_step)
        return {'swFCD': result}


@nb.njit
def _compute_swfcd_optimized(s, n_rois, t_max, window_size, window_step):
    """
    Optimized sliding window FCD computation using numba.
    """
    last_window = t_max - window_size
    N_windows = calc_length_numba(0, last_window, window_step)
    
    # Pre-compute all correlation matrices to avoid recomputation
    correlation_matrices = np.zeros((N_windows, n_rois, n_rois))
    window_idx = 0
    
    # Compute correlation matrices for all sliding windows
    for t in range(0, last_window, window_step):
        # Extract sliding window: shape (window_size+1, n_rois)
        window_data = np.zeros((window_size + 1, n_rois))
        for i in range(window_size + 1):
            for j in range(n_rois):
                window_data[i, j] = s[j, t + i]
        
        # Compute correlation matrix for this window
        corr_matrix = corrcoef_numba(window_data)
        correlation_matrices[window_idx] = corr_matrix
        window_idx += 1
    
    # Pre-extract all lower triangular parts to avoid recomputation
    n_pairs = int(n_rois * (n_rois - 1) / 2)
    lower_triangular_parts = np.zeros((N_windows, n_pairs))
    
    for i in range(N_windows):
        lower_triangular_parts[i] = extract_lower_triangular_numba(correlation_matrices[i], n_rois)
    
    # Compute correlations between all pairs of lower triangular parts
    cotsampling = np.zeros(int(N_windows * (N_windows - 1) / 2))
    
    kk = 0
    for ii2 in range(N_windows):
        for jj2 in range(ii2 + 1, N_windows):
            # Compute correlation between the two lower triangular parts
            ca = pearson_r_numba(lower_triangular_parts[ii2], lower_triangular_parts[jj2])
            cotsampling[kk] = ca
            kk += 1
    
    return cotsampling

    # ==================================================================
    # buildFullMatrix: given the output of from_fMRI, this function
    # returns the full matrix. Not needed, except for plotting and such...
    # ==================================================================
    def build_full_matrix(self, FCD_data):
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