from enum import IntEnum

import numpy as np
from numba import int32, float64, njit

from neuronumba.numba_tools.types import ArrF8_1d, ArrF8_2d

t_glu = 0
t_gaba = 1
We = 2
Wi = 3
I0 = 4
w = 5
J_NMDA = 6
M_e = 7
ae = 8
be = 9
de = 10
M_i = 11
ai = 12
bi = 13
di = 14
alfa_i = 15
alfa_e = 16
B_e = 17
B_i = 18
gamma = 19
rho = 20


class Parameters(IntEnum):
    t_glu = 0
    t_gaba = 1
    We = 2
    Wi = 3
    I0 = 4
    w = 5
    J_NMDA = 6
    M_e = 7
    ae = 8
    be = 9
    de = 10
    M_i = 11
    ai = 12
    bi = 13
    di = 14
    alfa_i = 15
    alfa_e = 16
    B_e = 17
    B_i = 18
    gamma = 19
    rho = 20

@njit(float64[:,:](float64[:], float64[:,:], float64[:,:]))
def Naskar_dfun(m: ArrF8_1d, state: ArrF8_2d, coupling: ArrF8_2d):
    Se = state[0, :]
    Si = state[1, :]
    J = state[2, :]
    Se = np.clip(Se, 0.0,1.0)
    Si = np.clip(Si, 0.0, 1.0)

    Ie = m[We] * m[I0] + m[w] * m[J_NMDA] * Se + m[J_NMDA] * coupling[0, :] - J * Si  # Eq for I^E (5). I_external = 0 => resting state condition.
    Ii = m[Wi] * m[I0] + m[J_NMDA] * Se - Si  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    y = m[M_e] * (m[ae] * Ie - m[be])
    re = y / (1.0 - np.exp(-m[de] * y))
    y = m[M_i] * (m[ai] * Ii - m[bi])
    ri = y / (1.0 - np.exp(-m[di] * y))
    dSe = -Se * m[B_e] + m[alfa_e] * m[t_glu] * (1. - Se) * re / 1000.  # divide by 1000 because we need milliseconds!
    dSi = -Si * m[B_i] + m[alfa_i] * m[t_gaba] * (1. - Si) * ri / 1000.
    dJ = m[gamma] * ri / 1000. * (re - m[rho]) / 1000.  # local inhibitory plasticity
    return np.stack((dSe, dSi, dJ))


class Naskar(object):

    n_params = len(Parameters)
    state_vars = ["S_e", "S_i", "J"]
    n_state_vars = len(state_vars)

    def __init__(self, **kwargs):
        self.m = np.empty(Naskar.n_params)
        self.m[t_glu] = 7.46
        self.m[t_gaba] = 1.82
        self.m[We] = 1.0
        self.m[Wi] = 0.7
        self.m[I0] = 0.382
        self.m[w] = 1.4
        self.m[J_NMDA] = 0.15
        self.m[M_e] = 1.0
        self.m[ae] = 310.0
        self.m[be] = 125.0
        self.m[de] = 0.16
        self.m[ai] = 615.0
        self.m[bi] = 177.0
        self.m[di] = 0.087
        self.m[M_i] = 1.0
        self.m[alfa_e] = 0.072
        self.m[alfa_i] = 0.53
        self.m[B_e] = 0.0066
        self.m[B_i] = 0.18
        self.m[gamma] = 1.0
        self.m[rho] = 3.0
        self.configure(**kwargs)

    def configure(self, **kwargs):
        params = list(Parameters)
        for k, v in kwargs.items():
            if k in params:
                self.m[Parameters[k]] = v

    def initial_state(self, n_rois):
        state = np.empty((Naskar.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        state[2] = 1.0
        return state

    @property
    def dfun(self):
        return Naskar_dfun

    @property
    def data(self):
        return self.m


