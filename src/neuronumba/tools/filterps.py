# --------------------------------------------------------------------------
# COMPUTE POWER SPECTRA FOR
# NARROWLY FILTERED DATA WITH LOW BANDPASS (0.04 to 0.07 Hz)
# not # WIDELY FILTERED DATA (0.04 Hz to justBelowNyquistFrequency)
#     # [justBelowNyquistFrequency depends on TR,
#     # for a TR of 2s this is 0.249 Hz]
# #--------------------------------------------------------------------------
import numpy as np
from scipy import stats
from enum import Enum
from neuronumba.tools.filters import BandPassFilter

class FiltPowSpetraVersion(Enum):
    v2021 = "v2021" # This is now the default one (from Irene's code)
    v2015 = "v2015" # This was the original version implemented (from Victor Saenger code)

def conv(u: np.ndarray, v: np.ndarray):  # python equivalent to matlab conv 'same' method
    # from https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
    npad = len(v) - 1
    full = np.convolve(u, v, 'full')
    first = npad - npad // 2
    return full[first:first + len(u)]


def gaussfilt(
    t: np.ndarray, 
    z: np.ndarray, 
    sigma: float
):
    # Apply a Gaussian filter to a time series
    #    Inputs: t = independent variable, z = data at points t, and
    #        sigma = standard deviation of Gaussian filter to be applied.
    #    Outputs: zfilt = filtered data.
    #
    #    based on the code by James Conder. Aug 22, 2013
    #    (partial) translation by Gustavo Patow
    n = z.size  # number of data
    a = 1 / (np.sqrt(2 * np.pi) * sigma)  # height of Gaussian
    sigma2 = sigma * sigma

    # check for uniform spacing
    # if so, use convolution. if not use numerical integration
    # uniform = false;
    dt = np.diff(t)
    dt = dt[0]
    # ddiff = max(abs(diff(diff(t))));
    # if ddiff/dt < 1.e-4
    #     uniform = true;
    # end

    # Only the uniform option is considered
    filter = dt * a * np.exp(-0.5 * ((t - np.mean(t)) ** 2) / sigma2)
    i = filter < dt * a * 1.e-6
    filter = np.delete(filter, i)  # filter[i] = []
    zfilt = conv(z, filter)
    onesToFilt = np.ones(np.size(z))  # remove edge effect from conv
    onesFilt = conv(onesToFilt, filter)
    zfilt = zfilt / onesFilt

    return zfilt

def filt_pow_spetra(
    signal: np.ndarray, 
    TR: float, 
    bpf: BandPassFilter, 
    version: FiltPowSpetraVersion = FiltPowSpetraVersion.v2021
):
    """
    signal: Time series of shape (time, regions)
    TR: Repetition time
    bpf: Bandpass filter
    """
    
    tmax, nNodes = signal.shape  # Updated shape to (time, regions)
    
    # Apply bandpass filtering (signal is now time x regions)
    ts_filt_narrow = bpf.filter(signal)
    
    # Perform FFT along time dimension (axis=0)
    if version == FiltPowSpetraVersion.v2021:
        pw_filt_narrow = np.abs(np.fft.fft(stats.zscore(ts_filt_narrow, axis=0), axis=0))
    elif version == FiltPowSpetraVersion.v2015:
        pw_filt_narrow = np.abs(np.fft.fft(ts_filt_narrow, axis=0))
    else:
        raise ValueError("Unknown version parameter")
    
    # Get the power spectrum for the first half of frequencies
    # We take slices up to tmax/2 from the first dimension (time)
    PowSpect_filt_narrow = pw_filt_narrow[0:int(np.floor(tmax / 2)), :] ** 2 / (tmax / (TR / 1000.0))
    
    return PowSpect_filt_narrow

def filt_pow_spetra_multiple_subjects(
    signal: np.ndarray, 
    tr: float, 
    bpf: BandPassFilter,
    version: FiltPowSpetraVersion=FiltPowSpetraVersion.v2021
):
    signal_array = None

    if type(signal) is dict:
        n_subjects = len(signal.keys())
        tmax, n_nodes = next(iter(signal.values())).shape
        # Convert dict to array
        signal_array = np.zeros((n_subjects, tmax, n_nodes))
        for i, s in enumerate(signal.keys()):
            signal_array[i] = signal[s]
    elif signal.ndim == 2:
        n_subjects = 1
        tmax, n_nodes = signal.shape
        signal_array = signal.reshape(1, tmax, n_nodes)
    else:
        n_subjects, tmax, n_nodes = signal.shape
        signal_array = signal
    
    Ts = tmax * (tr / 1000.0)
    freqs = np.arange(0, tmax / 2 - 1) / Ts

    if version == FiltPowSpetraVersion.v2021:
        PowSpect_filt_narrow = np.zeros((n_subjects, int(np.floor(tmax / 2)), n_nodes))
        f_diff_sub = np.zeros((n_subjects, n_nodes))

        for s in range(n_subjects):
            PowSpect_filt_narrow[s] = filt_pow_spetra(signal_array[s, :, :], tr, bpf, version)
            pow_areas = []
            for node in range(n_nodes):
                pow_areas.append(gaussfilt(freqs, PowSpect_filt_narrow[s][:, node], 0.005))
            pow_areas = np.array(pow_areas)
            max_idx = np.argmax(pow_areas, axis=1)
            f_diff_sub[s] = freqs[max_idx]
        
        f_diff = np.mean(f_diff_sub, axis=0)
        return f_diff
    
    elif version == FiltPowSpetraVersion.v2015:
        PowSpect_filt_narrow = np.zeros((n_subjects, int(np.floor(tmax / 2)), n_nodes))
        for s in range(n_subjects):
            PowSpect_filt_narrow[s] = filt_pow_spetra(signal[s, :, :], tr, bpf, version)
        Power_Areas_filt_narrow_unsmoothed = np.mean(PowSpect_filt_narrow, axis=0)  # (freqs, regions)

        Power_Areas_filt_narrow_smoothed = np.zeros_like(Power_Areas_filt_narrow_unsmoothed)
        for seed in range(n_nodes):
            Power_Areas_filt_narrow_smoothed[:, seed] = gaussfilt(freqs, Power_Areas_filt_narrow_unsmoothed[:, seed], 0.01)
        
        idxFreqOfMaxPwr = np.argmax(Power_Areas_filt_narrow_smoothed, axis=0)
        f_diff = freqs[idxFreqOfMaxPwr]
        return f_diff
    
    else:
        raise ValueError("Unknown version parameter")