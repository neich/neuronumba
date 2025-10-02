import numpy as np
from scipy.signal import butter, detrend, filtfilt

from neuronumba.basic.attr import HasAttr, Attr


class BandPassFilter(HasAttr):

    k = Attr(default=2, required=False, doc="Filter order")
    tr = Attr(default=None, required=True, doc="Repetition time in milliseconds")
    flp = Attr(default=None, required=True, doc="Low cut-off frequency in Hz")
    fhi = Attr(default=None, required=True, doc="High cut-off frequency in Hz")
    remove_artifacts = Attr(default=True, required=False, doc="Remove artifacts")

    apply_demean = Attr(default=True, required=False)
    apply_detrend = Attr(default=True, required=False)
    apply_finalDetrend = Attr(default=False, required=False)

    def filter(self, signal):
        """

        :param signal: signal to filter with shape (n_time_samples, n_rois)
        :return:
        """
        t_max, n_rois = signal.shape
        # Convert to seconds (self.tr units are ms)
        tr = self.tr / 1000.0
        fnq = 1. / (2. * tr)  # Nyquist frequency
        Wn = [self.flp / fnq, self.fhi / fnq]  # butterworth bandpass non-dimensional frequency
        bfilt, afilt = butter(self.k, Wn, btype='band', analog=False, output='ba')  # construct the filter
        # bfilt = bfilt_afilt[0]; afilt = bfilt_afilt[1]  # numba doesn't like unpacking...
        signal_filt = np.empty(signal.shape)
        for n in range(n_rois):
            if not np.isnan(signal[:, n]).any():
                ts = detrend(signal[:, n]) if self.apply_detrend else signal[:, n]
                ts = ts - np.mean(ts) if self.apply_demean else ts

                if self.remove_artifacts:
                    ts[ts > 3. * np.std(ts)] = 3. * np.std(ts)  # Remove strong artefacts
                    ts[ts < -3. * np.std(ts)] = -3. * np.std(ts)  # Remove strong artefacts

                signal_filt[:, n] = filtfilt(bfilt, afilt, ts, padlen=3 * (max(len(bfilt),
                                                                                  len(afilt)) - 1))  # Band pass filter. padlen modified to get the same result as in Matlab

                signal_filt[:, n] = detrend(signal_filt[:, n]) if self.apply_finalDetrend else signal_filt[:, n]

            else:  # We've found problems, mark this region as "problematic", to say the least...
                raise FloatingPointError("NaN found when applying BandPassFilter!")

        return signal_filt