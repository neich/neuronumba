from numba import njit

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.observables import measures
from neuronumba.observables.phase_interaction_matrix import phase_interaction_matrix
import numpy as np


class PhFCD(HasAttr):

    discard_offset = Attr(default=10, required=False)

    def from_fmri(self, ts):  # Compute the FCD of an input BOLD signal
        pim_matrix = phase_interaction_matrix(ts)  # Compute the Phase-Interaction Matrix
        return PhFCD_from_fmri(ts, self.discard_offset, pim_matrix)

@njit
def PhFCD_from_fmri(ts, discard_offset, pim_matrix):
    (t_max, n_rois) = ts.shape
    # calculates the size of phfcd vector
    npattmax = t_max - (2 * discard_offset - 1)
    # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
    size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)
    # Indices of triangular lower part of matrix
    # Isubdiag = tril_indices_column(n_rois, k=-1)
    # The int() is not needed, but... (see above)
    phIntMatr_upTri = np.zeros((npattmax, int(n_rois * (n_rois - 1) / 2)))
    for t in range(npattmax):
        i = 0
        for r in range(n_rois):
            for c in range(r+1, n_rois):
                phIntMatr_upTri[t, i] = pim_matrix[t, r, c]
                i += 1

    phfcd = numba_phFCD(phIntMatr_upTri, npattmax, size_kk3)

    return phfcd

@njit
def numba_phFCD(pim_up_tri, npattmax, size_kk3):
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