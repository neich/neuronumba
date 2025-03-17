import numpy as np
from neuronumba.basic.attr import HasAttr, Attr

class Observable(HasAttr):
    """
    Abstract class for Observables. Each implementation has to define "_compute" method.
    Inputs are passed as attributes to each observable implementation.
    Returns a dictionary with the results or None on error.
    """

    def compute(self):
        return self._compute()

    def _compute(self):
        raise NotImplemented('Should have been implemented by subclass!')


class ObservableFMRI(Observable):
    """
    This class is used to mantain backwards-compatibilty with previous implementations.
    Now the bold_signal input parameter is stored as an attribute but still maintains the
    "from_fmri" function entry point.

    It can be called in both styles:
        - As a generic Observable:
            obs = FooObsFMRI()
            obs.bold_singal = signal
            result = obs.compute()
        - Backwards-compatibility:
            obs = FooObsFMRI()
            result = obs.from_fmri(signal)
    """

    ignore_nans = Attr(default=False)
    # bold_signal (ndarray): Bold signal with shape (n_time_samples, n_rois)
    bold_signal = Attr(default=None)

    def from_fmri(self, bold_signal):
        self.bold_signal = bold_signal
        return self.compute()

    def _compute(self):
        if self.bold_signal is None or not (isinstance(self.bold_signal, np.ndarray) and self.bold_signal.ndim == 2):
            raise TypeError("Invalid bold signal.")
        if not self.ignore_nans and np.isnan(self.bold_signal).any():
            # TODO: Maybe we should raise an error?
            return np.nan
        return self._compute_from_fmri(self.bold_signal)

    def from_surrogate(self, bold_signal):
        t_max, n_parcells = bold_signal.shape
        for seed in range(n_parcells):
            bold_signal[:, seed] = bold_signal[np.random.permutation(t_max), seed]
        bold_su = bold_signal[:, np.random.permutation(n_parcells)]
        return self.from_fmri(bold_su)

    def _compute_from_fmri(self, bold_signal):
        raise NotImplemented('Should have been implemented by subclass!')