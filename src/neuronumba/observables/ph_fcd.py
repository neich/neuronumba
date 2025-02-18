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
        pim_matrix = phase_interaction_matrix(s)  # Compute the Phase-Interaction Matrix
        return PhFCD_from_fmri(n_rois, t_max, self.discard_offset, pim_matrix)

    # ==================================================================
    # buildFullMatrix: given the output of from_fMRI, this function
    # returns the full matrix. Not needed, except for plotting and such...
    # ==================================================================
    def buildFullMatrix(self, FCD_data):
        LL = FCD_data.shape[0]
        # T is size of the matrix given the length of the lower/upper triangular part (displaced by 1)
        T = int((1. + np.sqrt(1. + 8. * LL)) / 2.)
        fcd_mat = np.zeros((T, T))
        fcd_mat[np.triu_indices(T, k=1)] = FCD_data
        fcd_mat += fcd_mat.T
        return fcd_mat


def PhFCD_from_fmri(n_rois, t_max, discard_offset, pim_matrix):
    # calculates the size of phfcd vector
    npattmax = t_max - (2 * discard_offset - 1)
    # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
    size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)
    # Indices of triangular lower part of matrix
    # Isubdiag = tril_indices_column(n_rois, k=-1)
    # The int() is not needed, but... (see above)
    pim_up_tri = np.zeros((npattmax, int(n_rois * (n_rois - 1) / 2)))
    row_i, col_i = np.nonzero(np.tril(np.ones(n_rois), k=-1).T)  # Matlab works in column-major order, while Numpy works in row-major.
    i_subdiag = (col_i, row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    for t in range(npattmax):
        pim_up_tri[t, :] = pim_matrix[t][i_subdiag]

    return {'phFCD': PhFCD_from_fmri_numba(size_kk3, npattmax, pim_up_tri)}


@nb.njit(nb.f8[:](nb.intc, nb.intc, nb.f8[:, :]))
def PhFCD_from_fmri_numba(size_kk3, npattmax, pim_up_tri):
    phfcd = np.zeros((size_kk3))
    kk3 = 0

    for t in range(npattmax - 2):
        p1_sum = np.sum(pim_up_tri[t:t + 3, :], axis=0)
        p1_norm = np.linalg.norm(p1_sum)
        for t2 in range(t + 1, npattmax - 2):
            p2_sum = np.sum(pim_up_tri[t2:t2 + 3, :], axis=0)
            p2_norm = np.linalg.norm(p2_sum)

            dot_product = np.dot(p1_sum, p2_sum)
            phfcd[kk3] = dot_product / (p1_norm * p2_norm)
            kk3 += 1

    return phfcd