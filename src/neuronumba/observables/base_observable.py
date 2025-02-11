import numpy as np
from neuronumba.basic.attr import HasAttr, Attr

class Observable(HasAttr):
    """
    Abstract class for Observables. Each implementation has to
    define "_compute" method.
    Inputs are passed as attributes to each observable implementation.
    """

    def compute(self):
        return self._compute()
    
    def _compute(self):
        raise NotImplemented('Should have been implemented by subclass!')

class ObservableFMRI(Observable):
    ignore_nans = Attr(default=False)

    # bold_signal (ndarray): Bold signal with shape (n_rois, n_time_samples)
    bold_signal = Attr(default=None)

    def from_fmri(self, bold_signal):
        self.bold_signal = bold_signal
        return self.compute()

    def _compute(self):
        if self.bold_signal is None or not (isinstance(self.bold_signal, np.ndarray) and self.bold_signal.ndim == 2):
            raise TypeError("Invalid bold signal.")
        return self._compute_from_fmri(self.bold_signal)
    
    def from_surrogate(self, bold_signal):
        n_parcells, t_max = bold_signal.shape
        for seed in range(n_parcells):
            bold_signal[seed, :] = bold_signal[seed, np.random.permutation(t_max)]
        bold_su = bold_signal[np.random.permutation(n_parcells), :]
        return self.from_fmri(bold_su)

    def _compute_from_fmri(self, bold_signal):
        raise NotImplemented('Should have been implemented by subclass!')