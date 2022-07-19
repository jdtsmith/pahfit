"""Parameter structure for PAHFIT model fit"""

import numpy as np
from const import geometry
from pahfit.errors import PAHFITModelError
from pahfit.util import bounded_is_fixed, bounded_is_missing
from .pfnumba import using_numba, pahfit_jit, jitclass

feature_dtype = np.dtype([('const_prof', np.int32), ('nvranges', np.int32)])
params_dtype = np.dtype([('type', np.int32), ('ind', np.int32)])
tied_dtype = np.dtype([('ind', np.int32), ('start', np.int32),
                       ('count_num', np.int32), ('count_denom', np.int32),
                       ('sum_num', np.float64), ('sum_denom', np.float64)])


if using_numba:
    # jitclass typing
    feat_spec = dict(starlight=i4, dust_continuum=i4,n_mbb=i4,line=i4,
                     dust_features=i4,attenuation=i4, absorption=i4)
    param_spec = dict(y=f8[:], wavelength=f8[:],
                      bounds_low=i4[:], bounds_high=i4[:],
                      features=numba.from_dtype(feature_dtype)[:],
                      validity=i4[:],
                      params=numba.from_dtype(params_dtype)[:],
                      fixed=i4[:],
                      tied=numba.from_dtype(tied_dtype)[:],
                      tie_groups=i4[:],
                      const_profile=f8[:],
                      atten_geom = i4,
                      atten_curve = f8[:])
else:
    param_spec, feat_spec = (None, None)
    



@jitclass(feat_spec)
class FeatureCount:
    def __init__(self):
        self.starlight = 0
        self.dust_continuum = 0
        self.n_mbb = 0
        self.line = 0
        self.dust_features = 0
        self.attenuation = 0
        self.absorption = 0

@jitclass(param_spec)
class PAHFITParams:
    """The PAHFIT parameter class.  An object of this class can be
    used internally by pahfit.model in the scipy.least_square model
    function for dynamical model evaluation.  Supports independent and
    fixed features, validity ranges, parameter bounds, and various
    types of feature ties.

    """
    
    feature_count: FeatureCount
    
    def __init__(self, m, n, n_feat, n_val, n_param, n_fixed,
                 n_tied=0, n_tie_groups=0, n_cp=0, atten=True,
                 atten_geom=geometry.mixed):
        self.feature_count = FeatureCount()
        if atten:
            self.feature_count.attenuation = 1
        self.y = np.empty(m, dtype=np.float64)
        self.wavelength = np.empty(m, dtype=np.float64)
        self.bounds_low = np.empty(n, dtype=np.float64)
        self.bounds_high = np.empty(n, dtype=np.float64)
        self.features = np.empty(n_feat, dtype=feature_dtype)
        self.validity = np.empty(n_val, dtype=np.int32)
        self.params = np.empty(n_param, dtype=params_dtype)
        self.fixed = np.empty(n_fixed, dtype=np.int32)
        if n_tied>0:
            self.tied = np.empty(n_param, dtype=tied_dtype)
            self.tie_groups = np.empty(n_tie_groups, dtype=np.int32)
        if n_cp>0:
            self.const_profile = np.empty(n_cp, dtype=np.float64)
        self.atten_geom = atten_geom
        self.atten_curve = np.empty(m, dtype=np.float64)

    #def update(self, bounds_low, bounds_high):

    
@njit 
