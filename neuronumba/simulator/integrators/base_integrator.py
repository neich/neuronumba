from numba import float64, jit
from numba.experimental import jitclass


@jitclass
class Integrator:
    dt: float64

    def __init__(self, dt: float64):
        self.dt = dt


@jit(nopython=True)
def Integrator__init__(instance, dt):
    instance.dt = dt


