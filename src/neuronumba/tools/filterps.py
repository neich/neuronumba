# --------------------------------------------------------------------------
# COMPUTE POWER SPECTRA FOR
# NARROWLY FILTERED DATA WITH LOW BANDPASS (0.04 to 0.07 Hz)
# not # WIDELY FILTERED DATA (0.04 Hz to justBelowNyquistFrequency)
#     # [justBelowNyquistFrequency depends on TR,
#     # for a TR of 2s this is 0.249 Hz]
# #--------------------------------------------------------------------------
import numpy as np


def conv(u, v):  # python equivalent to matlab conv 'same' method
    # from https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
    npad = len(v) - 1
    full = np.convolve(u, v, 'full')
    first = npad - npad // 2
    return full[first:first + len(u)]


def gaussfilt(t, z, sigma):
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

def filt_pow_spetra(signal, TR, bpf):
    """
    signal: Time series of shape (time, regions)
    TR: Repetition time
    bpf: Bandpass filter
    """
    
    tmax, nNodes = signal.shape  # Updated shape to (time, regions)
    
    # Apply bandpass filtering (signal is now time x regions)
    ts_filt_narrow = bpf.filter(signal)
    
    # Perform FFT along time dimension (axis=0)
    pw_filt_narrow = np.abs(np.fft.fft(ts_filt_narrow, axis=0))
    
    # Get the power spectrum for the first half of frequencies
    # We take slices up to tmax/2 from the first dimension (time)
    PowSpect_filt_narrow = pw_filt_narrow[0:int(np.floor(tmax / 2)), :].T ** 2 / (tmax / (TR / 1000.0))
    
    return PowSpect_filt_narrow

def filt_pow_spetra_multiple_subjects(signal, tr, bpf):
    if type(signal) is dict:
        n_subjects = len(signal.keys())
        tmax, n_nodes = next(iter(signal.values())).shape
        PowSpect_filt_narrow = np.zeros((n_subjects, n_nodes, int(np.floor(tmax / 2))))
        for i, s in enumerate(signal.keys()):
            # Transpose signal to match expected input for filt_pow_spetra
            PowSpect_filt_narrow[i] = filt_pow_spetra(signal[s][:tmax, :], tr, bpf).T
        Power_Areas_filt_narrow_unsmoothed = np.mean(PowSpect_filt_narrow, axis=0).T
    elif signal.ndim == 2:
        n_subjects = 1
        tmax, n_nodes = signal.shape # Here we are assuming we receive only ONE subject...
        Power_Areas_filt_narrow_unsmoothed = filt_pow_spetra(signal, tr, bpf).T
    else:
        # In case we receive more than one subject, we do a mean...
        n_subjects, tmax, n_nodes = signal.shape
        PowSpect_filt_narrow = np.zeros((n_subjects, n_nodes, int(np.floor(tmax / 2))))
        for s in range(n_subjects):
            # Now signal shape is [subject, time, regions]
            PowSpect_filt_narrow[s] = filt_pow_spetra(signal[s, :, :], tr, bpf).T
        Power_Areas_filt_narrow_unsmoothed = np.mean(PowSpect_filt_narrow, axis=0).T
    
    Power_Areas_filt_narrow_smoothed = np.zeros_like(Power_Areas_filt_narrow_unsmoothed)
    Ts = tmax * (tr / 1000.0)
    freqs = np.arange(0, tmax / 2 - 1) / Ts
    for seed in np.arange(n_nodes):
        Power_Areas_filt_narrow_smoothed[:, seed] = gaussfilt(freqs, Power_Areas_filt_narrow_unsmoothed[:, seed], 0.01)
    
    # a-minimization seems to only work if we use the indices for frequency of
    # maximal power from the narrowband-smoothed data
    idxFreqOfMaxPwr = np.argmax(Power_Areas_filt_narrow_smoothed, axis=0)
    f_diff = freqs[idxFreqOfMaxPwr]
    return f_diff