import numba as nb
import numpy as np

if nb.config.DISABLE_JIT:
    def set_seed(seed):
        np.random.seed(seed)
else:
    def set_seed(seed):
        np.random.seed(seed)
        @nb.njit
        def numba_set_seed(value):
            np.random.seed(value)
        numba_set_seed(seed)
    