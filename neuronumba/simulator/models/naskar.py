import numpy as np
from numba import int32, float64
from numba.experimental import jitclass

from neuronumba.simulator.models.base_model import Model_init, Model_spec

spec = [
    ('t_glu', float64),
    ('t_gaba', float64),
    ('We', float64),
    ('I0', float64),
    ('w', float64),
    ('J_NMDA', float64),
    ('Wi', float64),
    ('M_e', float64),
    ('ae', float64),
    ('be', float64),
    ('de', float64),
    ('M_i', float64),
    ('ai', float64),
    ('bi', float64),
    ('di', float64),
    ('B_e', float64),
    ('alfa_i', float64),
    ('alfa_e', float64),
    ('B_i', float64),
    ('gamma', float64),
    ('rho', float64)
]


@jitclass(Model_spec + spec)
class Naskar(object):

    def __init__(self):
        Model_init(self, ["S_e", "S_i", "J"])

    def configure(self, **kwargs):
        for attr, value in kwargs.items():
            self.attr = value
        self._init_dependant()

    def initial(self, n_rois):
        return np.zeros((self.n_state_vars, n_rois))

    def _init_dependant(self):
        pass

    def dfun(self, state, coupling, I_external=0.0):
        Se = state[0, :]
        Si = state[1, :]
        J = state[2, :]
        Se = np.clip(Se, 0.0,1.0)
        Si = np.clip(Si, 0.0, 1.0)

        Ie = self.We * self.I0 + self.w * self.J_NMDA * Se + self.J_NMDA * coupling[0, :] - J * Si + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
        Ii = self.Wi * self.I0 + self.J_NMDA * Se - Si  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
        y = self.M_e * (self.ae * Ie - self.be)
        re = y / (1.0 - np.exp(-self.de * y))
        y = self.M_i * (self.ai * Ii - self.bi)
        ri = y / (1.0 - np.exp(-self.di * y))
        dSe = -Se * self.B_e + self.alfa_e * self.t_glu * (1. - Se) * re / 1000.  # divide by 1000 because we need milliseconds!
        dSi = -Si * self.B_i + self.alfa_i * self.t_gaba * (1. - Si) * ri / 1000.
        dJ = self.gamma * ri / 1000. * (re - self.rho) / 1000.  # local inhibitory plasticity
        return np.stack((dSe, dSi, dJ)), np.stack((Ie, re, J))

