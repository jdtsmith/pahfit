"""pahfit.util General pahfit utility functions."""
import numpy.ma as ma


def bounded_is_missing(val):
    """Return a mask array indicating which of the bounded values
    are missing.  A missing bounded value has a masked value."""
    return ma.getmask(val)[..., 0]


def bounded_is_fixed(val):
    """Return a mask array indicating which of the bounded values
    are fixed.  A fixed bounded value has masked bounds."""
    return ma.getmask(val)[..., -1].all(-1)
