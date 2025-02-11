import numpy as np
# import numba as nb

from neuronumba.observables.base_observable import ObservableFMRI
# from neuronumba.basic.attr import Attr


class FC(ObservableFMRI):
    """
    Main FC class.

    Args:
        fmri (ndarray): Bold signal with shape (n_rois, n_time_samples)
    """

    def _compute_from_fmri(self, fmri):
        cc = np.corrcoef(fmri.T, rowvar=False)  # Pearson correlation coefficients
        return {'FC': cc}
