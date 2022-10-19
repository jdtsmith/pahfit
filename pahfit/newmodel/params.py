# * pahfit.model.params
"""Parameter structure for PAHFIT model fit"""

# https://numba.discourse.group/t/best-practices-for-complex-structured-input/1406

import numpy as np
from const import geometry
from pahfit.features import Features
from pahfit.features.util import (bounded_is_fixed, bounded_is_missing,
                                  bounded_min, bounded_max)
from pahfit.packs.instrument import fwhm, within_segment, check_range
from pahfit.errors import PAHFITModelError
from .pfnumba import jitclass, pahfit_jit, using_numba
from .const import validity_gaussian_fwhms

# * Variables
TYPE_INDEP = -2
TYPE_FIXED = -1

_feature_dtype = np.dtype([('const_prof', np.int32), ('nvranges', np.int32)])
_params_dtype = np.dtype([('type', np.int32), ('ind', np.int32)])
_tied_dtype = np.dtype([('ind', np.int32), ('start', np.int32),
                        ('count_num', np.int32), ('count_denom', np.int32),
                        ('sum_num', np.float64), ('sum_denom', np.float64)])

if using_numba:
    # set up jitclass typing
    from numba import i4, f8, from_dtype
    _feat_spec = dict(starlight=i4, dust_continuum=i4, line=i4,
                      dust_features=i4, attenuation=i4, absorption=i4)
    _param_spec = dict(y=f8[:], wavelength=f8[:],
                       bounds_low=i4[:], bounds_high=i4[:],
                       features=from_dtype(_feature_dtype)[:],
                       validity=i4[:],
                       params=from_dtype(_params_dtype)[:],
                       fixed=i4[:],
                       tied=from_dtype(_tied_dtype)[:],
                       tie_groups=i4[:],
                       const_profile=f8[:],
                       atten_geom=i4,
                       atten_curve=f8[:], atten_vec=f8[:])
else:
    _feat_spec, _param_spec = (None, None)


# * Feature classes
@jitclass(_feat_spec)
class FeatureCount:
    def __init__(self):
        self.starlight = 0
        self.dust_continuum = 0
        self.line = 0
        self.dust_features = 0
        self.attenuation = 0
        self.absorption = 0


@jitclass(_param_spec)
class PAHFITParams:
    """The PAHFIT parameter class.  An object of this class is used
    internally by pahfit.model in the scipy.least_square model
    function for dynamic model evaluation.  Supports independent and
    fixed features, validity ranges, parameter bounds, and various
    types of feature ties (aka constraints), with and without
    renormalization.


    :class:PAHFITParams provides all the auxiliary parameter and
    constraint information needed for efficient and flexible
    calculation and optimization of the PAHFIT model.
    """
    feature_count: FeatureCount

    def __init__(self, m, n, n_feat, n_valid, n_param, n_fixed,
                 n_tied=0, n_tie_groups=0, n_cp=0, atten=True,
                 atten_geom=geometry.mixed):
        """Create a new PAHFITParams object, passing the number of
        data values, independent parameters, features, validity
        ranges, total parameters (included fixed and tied), fixed
        parameters, ties, tie groups, and constant profile features.
        Provide information on attenuation and the attenuation model.
        """
        self.feature_count = FeatureCount()

        self.n_wav = m

        self.y = np.empty(m, dtype=np.float64)
        self.wavelength = np.empty(m, dtype=np.float64)
        self.bounds_low = np.empty(n, dtype=np.float64)
        self.bounds_high = np.empty(n, dtype=np.float64)

        # Parameter lists:
        self.features = np.empty(n_feat, dtype=_feature_dtype)  # F-list
        self.validity = np.empty(n_valid, dtype=np.int32)       # V-list
        self.params = np.empty(n_param, dtype=_params_dtype)    # P-list
        self.fixed = np.empty(n_fixed, dtype=np.int32)          # X-list

        # Constraints
        if n_tied > 0:
            self.tied = np.empty(n_param, dtype=_tied_dtype)          # T-list
            self.tie_groups = np.empty(n_tie_groups, dtype=np.int32)  # G-list

        # Constant Profile features
        if n_cp > 0:
            self.const_profile = np.empty(n_cp, dtype=np.float64)  # C-list

        # Attenuation
        if atten:
            self.atten_geom = atten_geom
            self.atten_curve = np.empty(m, dtype=np.float64)  # A-list

    def get_par(self, ind, indep):
        """Return the fixed or independent parameter value of index IND."""
        par = self.params[ind]
        if par['type'] == TYPE_FIXED:
            return self.fixed[par['ind']]
        elif par['type'] == TYPE_INDEP:
            return indep[par['ind']]

    def compute_tie_sums(self, indep):
        """Recompute numerator and denominator sums for all
        non-trivial ties.  Note that one-to-many ratio constraints
        only require (or use) a denominator sum.
        """
        for i in range(self.tied.size):
            start = self.tied[i]['start']
            if self.tied[i]['count_num'] > 1:
                self.tied[i]['sum_num'] = 0.0
                for p in range(start, start + self.tied[i]['count_num']):
                    self.tied[i]['sum_num'] += self.get_par(self.tie_groups[p], indep)
                start += self.tied[i]['count_num']  # Move on to denominators
            self.tied[i]['sum_denom'] = 0.0
            for p in range(start, start + self.tied[i]['count_denom']):
                self.tied[i]['sum_denom'] += self.get_par(self.tie_groups[p], indep)

    def reset(self, indep):
        """Reset the internal state for a new calculation."""
        self.y[:] = 0.0
        self.compute_tie_sums(indep)
        self.f_off, self.p_off, self.v_off = 0, 0, 0

    def retrieve_param(self, indep):
        """Return the value of the next parameter on the parameter list.

        Parameters
        ----------

        INDEP: Current independent parameter vector.

        Retrieved parameters can be fixed, independent (varying), or
        tied via a ratio to some other parameter value(s).  They may
        also be subject to numerator normalization for many-to-one or
        many-to-many ties.
        """
        par = self.params[self.p_off]
        if par['type'] < 0:  # normal fixed or independent param (including trivial ties)
            val = self.get_par(self.p_off, indep)
        else:  # Non-trivial parameter tie, must compute
            tnum_off = par['type']  # type is overloaded as a numerator offset
            trec = self.tied[par['ind']]  # index is into the T-vec
            ratio = self.get_par(trec['ind'], indep)  # the parameter is the *ratio*
            val = ratio * trec['sum_denom']  # simple ratio tie
            if trec['count_num'] > 1:  # band-sum many-to-* tie: renormalize
                this_num = self.get_par(self.tie_groups[trec['start'] + tnum_off], indep)
                val *= this_num / trec['sum_num']
        self.p_off += 1
        return val

    def retrieve_params(self, indep, np):
        """Retrieve NP parameters, as a tuple.  See `retrieve_param'."""
        return tuple(self.retrieve_param(indep) for _ in range(np))

    def next_feature(self):
        self.f_off += 1

    def accumulate(self, indep, func, np, attenuation=False):
        """Accumulate a single model component feature with NP parameters.

        Parameters
        ----------

        INDEP: The independent parameter vector.

        FUNC: a function of wavelength and NP additional arguments
          which returns a model component vector (e.g. blackbody).

        NP: the number of additional parameters FUNC takes.  If the
          feature is marked as a "constant profile" features, this is
          ignored, and a single parameter (the current amplitude) is
          used.

        By default, accumulates the new model component into the
        (model) y-vector, unless ATTENUATION is True, in which case
        the attenuation-vector is used as the accumulation target.
        Increments the parameter and feature offsets.  Accumulates
        only over the valid range(s), if validity is set.
        """
        cp = self.features[self.f_off]['const_prof']
        nvr = self.features[self.f_off]['nvranges']
        self.f_off += 1
        vec = self.atten_curve if attenuation else self.y

        if cp > 0:  # Constant, pre-computed profile
            if self.params[self.p_off]['type'] == TYPE_FIXED:  # FIXED amplitude
                amp = 1.
                self.p_off += 1  # It's precomputed, skip it
            else:  # non-fixed: single parameter is an amplitude
                amp = self.retrieve_params(indep, 1) / self.const_profile[cp]
            if nvr == 0:  # Valid everywhere
                vec += amp * self.const_profile[cp + 1:cp + 1 + self.n_wav]
            else:
                coff = 0
                for _ in range(nvr):  # A set of constant profile features
                    low, high = self.validity[self.v_off:self.v_off + 2]
                    lv = high - low
                    vec[low:high] += amp * self.const_profile[cp + 1 + coff:cp + 1 + coff + lv]
                    coff += lv + 1
                    self.v_off += 2
        else:  # must compute fresh
            params = self.retrieve_params(indep, np)
            if nvr == 0:
                vec += func(self.wavelength, *params)
            else:
                coff = 0
                for _ in range(nvr):
                    low, high = self.validity[self.v_off:self.v_off + 2]
                    vec[low:high] += func(self.wavelength[low:high], *params)
                    self.v_off += 2


@pahfit_jit
def _pahfit_params(wavelength, features, params, indep, fixed,
                   atten_curve=None, atten_geom=None,
                   validity=None, const_profile=None,
                   tied=None, tie_groups=None):
    """Simple convenience wrapper to create a :class:PAHFITParams
    parameter map based from input np.ndarray's."""
    return PAHFITParams()  # This will cache the jitclass!

