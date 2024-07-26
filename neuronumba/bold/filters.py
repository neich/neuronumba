import numpy as np
from scipy.signal import butter, detrend, filtfilt

from neuronumba.basic.attr import HasAttr, Attr


class BandPassFilter(HasAttr):
    
    k = Attr(default=2, required=False)
    tr = Attr(default=None, required=True)
    flp = Attr(default=None, required=True)
    fhi = Attr(default=None, required=True)
    remove_artifacts = Attr(default=True, required=False)
    
    def filter(self, signal):
        """

        :param signal: signal to filter with shape (n_rois, n_time_samples)
        :return:
        """
        n_rois, t_max = signal.shape
        fnq = 1. / (2. * self.tr)  # Nyquist frequency
        Wn = [self.flp / fnq, self.fhi / fnq]  # butterworth bandpass non-dimensional frequency
        bfilt, afilt = butter(self.k, Wn, btype='band', analog=False, output='ba')  # construct the filter
        # bfilt = bfilt_afilt[0]; afilt = bfilt_afilt[1]  # numba doesn't like unpacking...
        signal_filt = np.empty(signal.shape)
        for seed in range(n_rois):
            if not np.isnan(signal[seed, :]).any():  # No problems, go ahead!!!
                ts = signal[seed, :] - np.mean(signal[seed, :])

                if self.remove_artifacts:
                    ts[ts > 3. * np.std(ts)] = 3. * np.std(ts)  # Remove strong artefacts
                    ts[ts < -3. * np.std(ts)] = -3. * np.std(ts)  # Remove strong artefacts

                signal_filt[seed, :] = filtfilt(bfilt, afilt, ts, padlen=3 * (max(len(bfilt),
                                                                                  len(afilt)) - 1))  # Band pass filter. padlen modified to get the same result as in Matlab

            else:  # We've found problems, mark this region as "problematic", to say the least...
                raise FloatingPointError("NaN found when applying BandPassFilter!")

        return signal_filt