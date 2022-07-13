"""pahfit.model.numba - Helper imports for the optional numba
dependency.  Use pf_njit as a function decorator.
"""
try:
    from numba import i4, f8, jit    # import the types
    from numba.experimental import jitclass
    using_numba = True
    def pahfit_jit(*args, **kwargs):
        return jit(*args, **kwargs, nopython=True, cache=True)
except ImportError:
    using_numba = False
    def pahfit_jit(*args, **kwargs):
        return lambda f: f
    jitclass = pahfit_jit

