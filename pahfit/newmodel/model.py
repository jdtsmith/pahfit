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

    param_map.reset(params)

    # Note bene: order is essential!
    # Starlight
    for _ in range(param_map.feature_count.starlight):
        param_map.accumulate(params, blackbody, 2)

    # Dust continuum
    for _ in range(param_map.feature_count.dust_continuum):
        param_map.accumulate(params, modified_blackbody, 2)

    # Lines
    for _ in range(param_map.feature_count.line):
        param_map.accumulate(params, gaussian, 3)

    # Dust features
    for _ in range(param_map.feature_count.dust_features):
        param_map.accumulate(params, drude, 3)

    # Attenuation
    if param_map.feature_count.attenuation:
        tau97 = param_map.retrieve_param(params)
        param_map.atten_vec[:] = tau97 * param_map.atten_curve

        # Absorption: add discrete absorption features with their own parameters
        for _ in range(param_map.feature_count.absorption):
            param_map.accumulate(params, drude, 3, attenuation=True)

        if param_map.atten_geom == geometry.mixed:
            param_map.y *= (1.0 - np.exp(-param_map.atten_vec)) / param_map.atten_vec
        else:  # screen
            param_map.y *= np.exp(-param_map.atten_vec)

    return param_map.y
