import numpy as np

from src.neuronumba import HasAttr


class ObservableAccumulator(HasAttr):
    def init(self, S, N):
        raise Exception('observable accumulator not defined')


class AveragingAccumulator(ObservableAccumulator):
    def init(self, S, N):
        return np.zeros((S, N, N))

    def accumulate(self, FCs, nsub, signal):
        FCs[nsub] = signal
        return FCs

    def postprocess(self, FCs):
        return np.squeeze(np.mean(FCs, axis=0))


class ConcatenatingAccumulator(ObservableAccumulator):
    def init(self, S, N):
        return np.array([], dtype=np.float64)

    def accumulate(self, FCDs, nsub, signal):
        FCDs = np.concatenate((FCDs, signal))  # Compute the FCD correlations
        return FCDs

    def postprocess(self, FCDs):
        return FCDs  # nothing to do here
