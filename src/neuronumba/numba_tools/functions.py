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


@nb.njit
def pearson_corr_numba_1d(x, y):
    n = x.size
    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    return num / np.sqrt(den_x * den_y)

@nb.njit
def pearson_corrcoef_numba_2d(X):
    n_vars, n_obs = X.shape
    result = np.empty((n_vars, n_vars), dtype=np.float64)
    means = np.empty(n_vars, dtype=np.float64)
    stds = np.empty(n_vars, dtype=np.float64)
    
    # Compute means and std deviations for each variable (row)
    for i in range(n_vars):
        sum_x = 0.0
        for k in range(n_obs):
            sum_x += X[i, k]
        means[i] = sum_x / n_obs
        
        sum_sq = 0.0
        for k in range(n_obs):
            dx = X[i, k] - means[i]
            sum_sq += dx * dx
        stds[i] = np.sqrt(sum_sq / n_obs)
    
    # Compute the correlation matrix
    for i in range(n_vars):
        for j in range(n_vars):
            num = 0.0
            for k in range(n_obs):
                num += (X[i, k] - means[i]) * (X[j, k] - means[j])
            denom = n_obs * stds[i] * stds[j]
            result[i, j] = num / denom if denom != 0 else 0.0
    return result