import numpy as np
from neuronumba.basic.attr import Attr
from typing import Union
from neuronumba.observables.base_observable import Observable
from neuronumba.tools import filterps

class HFreq(Observable):
    """
    Observable class to compute given a group, the h-frequencies for each node
    """

    # tr in ms
    tr = Attr(required=True, doc="TR time in ms")
    # The array or dictionary of the group that the frequencies will be extracted from. 
    # Each frmi on the array/dict should be in (time, nodes)
    group_fmri = Attr(default=None)
    # The version of filterps to use 
    filterps_version = Attr(default=filterps.FiltPowSpetraVersion.v2021)

    def _compute(self):
        if isinstance(self.tr, int):
            self.tr = float(self.tr)
        if not isinstance(self.tr, float):
            raise TypeError(f'Parameter "tr" must be float, got {type(self.tr).__name__}')
        if not isinstance(self.group_fmri, (np.ndarray, dict)):
            raise TypeError(f'Parameter "group_fmri" must be either np.ndarray or dict, got {type(self.group_fmri).__name__}')
        if not isinstance(self.filterps_version, filterps.FiltPowSpetraVersion):
            raise TypeError(f'Parameter "filterps_version" must be FilterPowSpetraVersion, got {type(self.filterps_version).__name__}')

        f_diff = filterps.filt_pow_spetra_multiple_subjects(self.group_fmri, self.tr, self.filterps_version)
        return 2 * np.pi * f_diff  # omega