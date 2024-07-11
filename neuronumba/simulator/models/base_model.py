from numba import types, typed, jit
from numba.experimental import jitclass

Model_spec = [
    ('state_vars', types.ListType(types.unicode_type)),
    ('n_state_vars', types.int32)
]


@jit(nopython=True)
def Model_init(instance, state_vars):
    instance.state_vars = typed.List(state_vars)
    instance.n_state_vars = len(state_vars)
