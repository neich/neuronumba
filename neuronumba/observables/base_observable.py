import numpy as np
from neuronumba.basic.attr import HasAttr, Attr


# Abstract class for Observables. At the moment it has a main method "from_fMRI" that takes the signal and the filter
# as parameters and outputs the result if computable (or None if some problem occurred). Each implementation has to
# define "_compute_from_fMRI" method.
#
# NOTES: Implementation is as this to maximize the portability with the old class based library.
class Observable(HasAttr):
    ignore_nans = Attr(default=False)
    
    # Main method to compute the Observable from an fMRI BOLD signal.
    def from_fmri(self, bold_signal):
        # ignoreNaN = 'ignore_nans' in kwargs and kwargs['ignore_nans']
        if not self.ignore_nans and np.isnan(bold_signal).any():
            return np.nan
        return self._compute_from_fmri(bold_signal)

    def from_surrogate(self, bold_signal):
        n_parcells, t_max = bold_signal.shape
        for seed in range(n_parcells):
            bold_signal[seed, :] = bold_signal[seed, np.random.permutation(t_max)]
        bold_su = bold_signal[np.random.permutation(n_parcells), :]
        return self.from_fmri(bold_su)

    def _compute_from_fmri(self, bold_signal):
        raise NotImplemented('Should have been implemented by subclass!')
