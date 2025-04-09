import numba as nb
import numpy.typing as npt
import numpy as np
import ctypes

@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """ returns a void pointer from a given memory address """
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

# The idea here is to be able to disable JIT compilation on numba so we can better debug.
# To do that we define two versions for this functions that go in-hand in code. 
# When JIT is disabled, then the buffer can be used directly, this is why get_addr return the same
# buffer, and create_carray does nothing more than return the "address", that it is the first value of get_addr return. 
if nb.config.DISABLE_JIT:
    def get_addr(a: npt.NDArray):
        return a, a.shape, a.dtype

    def create_carray(address, shape, dtype):
        return address
else:
    def get_addr(a: npt.NDArray):
        return a.ctypes.data, a.shape, a.dtype
    
    @nb.njit
    def create_carray(address: int, shape: tuple, dtype: np.dtype):
        return nb.carray(address_as_void_pointer(address), shape, dtype=dtype)