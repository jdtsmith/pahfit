"""
PAHFIT optimized model function.
"""

import numpy as np
from pahfit.newmodel.params import PAHFITParams
from .components import modified_blackbody, blackbody, drude, gaussian
from const import geometry


def pahfit_function(params, param_map: PAHFITParams):

    """Calculate and return the PAHFIT model function.

    Arguments
    ---------

    PARAMS: The 1D array of independent parameters, as passed by the
      optimizer.  Note that fixed parameters are not included (see
      FIXED, below).

    PARAM_MAP: A PAHFITParams object identifying all the
      independent, fixed, and tied parameters.
    """


    
