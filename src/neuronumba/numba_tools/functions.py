import math
import numba as nb
import numpy as np

@nb.njit(nb.f8[:](nb.f8[:]), cache=True)
def erfc_approx(x):
    # Coefficients for approximation
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    result = np.empty_like(x)
    for i in range(x.size):
        val = x[i]

        sign = 1
        if val < 0:
            sign = -1
        val = abs(val)

        t = 1.0 / (1.0 + p * val)
        y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        erf = 1.0 - y * math.exp(-val * val)
        result[i] = 1.0 - sign * erf

    return result

@nb.njit
def erfc_complex_array(z):
    # Coefficients for the approximation (Abramowitz & Stegun 7.1.26, adapted for complex)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    result = np.empty_like(z, dtype=np.complex128)
    for i in range(z.size):
        val = z.flat[i]
        sign = 1
        if val.real < 0:
            sign = -1
        absz = abs(val)
        t = 1.0 / (1.0 + p * absz)
        y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        erf = 1.0 - y * np.exp(-val * val)
        result.flat[i] = 1.0 - sign * erf
    return result
