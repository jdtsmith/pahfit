"""pahfit.model.numba - Helper imports for the optional numba
just-in-time compilation functionality of the PAHFIT model.  Use
pahfit_jit/jitclass as function/class decorators.
"""
# pyright: reportUnusedVariable=false

__all__ = ['pahfit_jit', 'jitclass']
from ..config import USE_NUMBA

try:
    if not USE_NUMBA:
        raise NotImplementedError
    from numba import jit
    from numba.experimental import jitclass
    using_numba = True
    def pahfit_jit(*args, **kwargs):
        return jit(*args, **kwargs, nopython=True, cache=True)
except (ImportError, NotImplementedError):
    using_numba = False
    def noop(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0] # simple decorator: return the function as is
        else:
            return lambda f: f # decorator with arguments: ignore them!
    pahfit_jit = noop
    jitclass = noop
