"""
PAHFIT optimized model function
"""

import numpy as np

from .components import blackbody, drude, gaussian, power, amplitude

# * Parameter Setup


    

def _feature_validity(lam0, fwhm, , drude=False):
    
                      
# ** Model Function

def pahfit_function(params, param_map):

    """Calculate and return the PAHFIT model function.

    Arguments
    ---------

    PARAMS: The 1D array of independent parameters, as passed by the
      optimizer.  Note that fixed parameters are not included (see
      FIXED, below).

    PARAM_MAP: An object identifying all the
      independent, fixed, and tied parameters.  This argument is
      automatically constructed by _create_param_map.  See the
      documentation for information on the internal details.
    """
    
    

    
