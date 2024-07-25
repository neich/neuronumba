import numpy as np
import numba as nb

from neuronumba.observables.base_observable import Observable
from neuronumba.basic.attr import Attr
from neuronumba.observables.phase_interaction_matrix import phase_interaction_matrix


class PhFCD(Observable):

    discard_offset = Attr(default=10, required=False)

    def _compute_from_fmri(self, bold_signal):  # Compute the FCD of an input BOLD signal
        pim_matrix = phase_interaction_matrix(bold_signal)  # Compute the Phase-Interaction Matrix
        return PhFCD_from_fmri(bold_signal.shape[0], bold_signal.shape[1], self.discard_offset, pim_matrix)


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

    return PhFCD_from_fmri_numba(size_kk3, npattmax, pim_up_tri)


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
