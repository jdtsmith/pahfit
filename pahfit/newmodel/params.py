# * pahfit.model.params
"""Parameter structure for PAHFIT model fit"""

# https://numba.discourse.group/t/best-practices-for-complex-structured-input/1406

import numpy as np
from const import geometry
from pahfit.features import Features
from pahfit.packs.instrument import fwhm, within_segment, check_range
from pahfit.errors import PAHFITModelError
from pahfit.util import bounded_is_fixed, bounded_is_missing, bounded_min, bounded_max
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
                       atten_geom = i4,
                       atten_curve = f8[:], atten_vec=f8[:])
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
    """The PAHFIT parameter class.  An object of this class can be
    used internally by pahfit.model in the scipy.least_square model
    function for dynamical model evaluation.  Supports independent and
    fixed features, validity ranges, parameter bounds, and various
    types of feature ties.
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

        if atten:
            self.feature_count.attenuation = 1
            self.atten_vec = np.empty(m, dtype=np.float64)
        self.y = np.empty(m, dtype=np.float64)
        self.wavelength = np.empty(m, dtype=np.float64)
        self.bounds_low = np.empty(n, dtype=np.float64)
        self.bounds_high = np.empty(n, dtype=np.float64)

        # Parameter lists:
        self.features = np.empty(n_feat, dtype=_feature_dtype) # F-list
        self.validity = np.empty(n_valid, dtype=np.int32)     # V-list
        self.params = np.empty(n_param, dtype=_params_dtype)   # P-list
        self.fixed = np.empty(n_fixed, dtype=np.int32)        # X-list

        # Constraints
        if n_tied>0:
            self.tied = np.empty(n_param, dtype=_tied_dtype)          # T-list
            self.tie_groups = np.empty(n_tie_groups, dtype=np.int32) # G-list

        # Constant Profile features
        if n_cp>0:
            self.const_profile = np.empty(n_cp, dtype=np.float64) # C-list

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
                start += self.tied[i]['count_num'] # Move on to denominators
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

        Parameters can be fixed, independent (varying), or tied via a
        ratio to some other parameter value(s).  They may also be
        subject to numerator normalization for many-to-one or
        many-to-many ties.  INDEP is the current independent parameter
        vector.
        """
        par = self.params[self.p_off]
        if par['type'] < 0: # normal fixed or independent param (including trivial ties)
            val = self.get_par(self.p_off, indep)
        else: # Non-trivial parameter tie, must compute
            tnum_off = par['type'] # type is overloaded as a numerator offset
            trec = self.tied[par['ind']] # index is into the T-vec
            ratio = self.get_par(trec['ind'], indep)  # ratio is the actual parameter
            val = ratio * trec['sum_denom']
            if trec['count_num'] > 1: # band-sum many-to-* tie: renormalize
                this_num = self.get_par(self.tie_groups[trec['start'] + tnum_off], indep)
                val *= this_num/trec['sum_num']
        self.p_off += 1
        return val

    def retrieve_params(self, indep, np):
        """Retrieve NP parameters, as a tuple.  See `retrieve_param'."""
        return tuple(self.retrieve_param(indep) for _ in range(np))
        
    def accumulate(self, indep, func, np, attenuation=False):
        """Accumulate a model component FUNC with NP parameters.

        FUNC is a function of wavelength and NP additional parameters
        which returns a model component vector (e.g. blackbody). By
        default, accumulates the new model component into the (model)
        y-vector, unless ATTENUATION is True, in which case the
        attenuation-vector is used as the target.
        """
        cp = self.features[self.f_off]['const_prof']
        nvr = self.features[self.f_off]['nvranges']

        vec = self.atten_vec if attenuation else self.y
        
        if cp > 0: # Constant, pre-computed profile
            if self.params[self.p_off]['type'] == TYPE_FIXED: # FIXED amplitude
                self.p_off += 1 # We don't actually need the amplitude value, skip it
                if nvr == 0:
                    vec += self.const_profile[cp+1:cp+1+self.n_wav]
                else:
                    coff = 0
                    for _ in range(nvr):
                        low, high = self.validity[self.v_off:self.v_off+2] 
                        lv = high - low
                        vec[low:high] += self.const_profile[cp+1+coff:cp+1+coff+lv]
                        coff += lv + 1
                        self.v_off += 2
            else: # non-fixed: parameter is a single amplitude
                amp = self.retrieve_params(indep, 1)
                amp /= self.const_profile[cp]
                if nvr == 0:
                    vec += amp * self.const_profile[cp+1:cp+1+self.n_wav]
                else:
                    coff = 0
                    for _ in range(nvr):
                        low, high = self.validity[self.v_off:self.v_off+2] # indices into y
                        lv = high - low # length of this validity chunk
                        vec[low:high] += amp * self.const_profile[cp+1+coff:cp+1+coff+lv]
                        coff += lv + 1
                        self.v_off += 2
        else:  # must compute fresh
            params = self.retrieve_params(indep, np)
            if nvr == 0:
                vec += func(self.wavelength, *params)
            else:
                coff = 0
                for _ in range(nvr):
                    low, high = self.validity[self.v_off:self.v_off+2]
                    vec[low:high] += func(self.wavelength[low:high], *params)
                    self.v_off += 2

# * Interface
@pahfit_jit
def build_params(features, spectra, fwhm_func, redshift = None):
    """Create and return a fully populated :class:PAHFITParams
    parameter map.

    :class:PAHFITParams provides all the auxiliary parameter and
    constraint information needed for efficient and flexible
    calculation and optimization of the PAHFIT model.

    Parameters
    ----------
    
    features : :class:`pahfit.features.Features` or str
        Either a Features table, or a string specifying one of the
        default science pack tables to read in as a Features table.

        The Features table contains information on all (groups of)
        features, their kinds, parameters (including both value and
        bounds), as well as any ties between parameters or parameter
        groups (in its meta-data).  See the documentation for
        information on the structure of this table.

    spectra : :class:`specutils.Spectrum1D` or iterable thereof
        A single instance or an iterable of (potentially
        wavelength-overlapping) specutils.Spectrum1D objects to fit.
        The canonical "instrument" name or names applicable for each
        spectrum must be encoded in each spectral segment's `meta' table
        dict under the key "instrument".  Redshift can also be set in
        the Spectrum1D object, or can be specified with a REDSHIFT
        argument; see below.

    fwhm_func : function(segment_name, wavelengths)
        A function of two arguments, which provides FWHM information
        from the instrument pack:

           fwhm = fwhm_func(instrument_segment, observed_frame_wavelength_microns)

        which returns the fwhm (in microns) based on segment
        instrument name(s) and the (observed frame) wavelength of
        the line, in microns.

        Note: Normally, line FWHM is considered a detail of the
        instrument pack, and never needs consideration.  If a line's
        FWHM is provided in advance in the features table (with or
        without bounds), it will OVERRIDE the value in the
        instrument pack.  Upon fit completion, line FWHM will ONLY
        be updated for output in the features table for any lines
        whose FWHM was provided in advance (and not fixed).

    redshift : float, optional
        The unitless redshift (z = delta(lam)/lam_0).  Assumed to be
        0.0 if omitted.  The passed redshift should be accurate to
        within a small fraction of the finest spectral resolution
        element in the passed collection of spectra.  Note that any
        redshift provided in individual spectra overrides the redshift
        (if any) passed via this argument.

    Returns
    -------

    param_map
        A populated :class:`PAHFITParams` object, which includes
        scalars and (structured) numpy arrays.  This structure is an
        internal implementation detail of the PAHFIT model, and not
        for external use.  In all cases the arrays are simple 1D
        arrays, either integer, double floating point, or named
        structured arrays of integers/doubles.  See the documentation
        for details on the various param_map arrays.

        Note: it is possible to re-use a param_map object between
        fits, assuming none of the model/parameter details have
        changed, other than starting parameter value(s)."""

    #pp = PAHFITParams(). # XXX maybe doesn't need to be separate. This will cache the jitclass!

    # TODO: Check cached param_map if passed.  What checks to be done?
    # Just opt-in and let the user be responsible.

    # *** Ingest Spectra
    if not isinstance(spectra, (tuple, list)):
        spectra = (spectra,)

    spec_samples = sum(len(s.spectral_axis) for s in spectra)


    # Flux and Wavelength
    flux = np.empty(spec_samples, dtype=np.float64)
    flux_unc = np.empty_like(flux)
    wavelength = np.empty_like(flux)
    off = 0
    waves = {}
    for i, sp in enumerate(spectra):
        w = sp.spectral_axis.micron

        try:
            ins = sp.meta['instrument'] # can be a list of strings or glob
        except KeyError:
            raise PAHFITModelError(f"Instrument missing from input spectrum #{i}.")
        n = len(w)
        bnds = [w[0], w[-1]] # Assume monotonic, either increasing or decreasing
        if bnds[0] > bnds[1]:
            bnds = bnds.reverse()

        check_range(bnds, ins) # Check for compatibility between passed wave-range and instrument
        z = sp.spectral_axis.redshift.value if redshift is None else redshift
        if ins in waves:
            raise PAHFITModelError(f"Instrument present in more than one input spectrum: {ins}")
            
        waves[ins]=dict(obs_wave = w, start = off, z = z, bounds = bnds)
        f = sp.flux
        f_unc = sp.uncertainty
        if f_unc is None:
            raise PAHFITModelError(f"Flux uncertainty is required: input spectrum: {ins}")

        # De-redshift
        if not z == 0.0:
            w /= (1. + z)
            f *= (1. + z)  # Conserve power!
            f_unc *= (1. + z)

        wavelength[off:off + n] = w
        flux[off:off + n] = f
        flux_unc[off:off + n] = f_unc

        off += n

    # Parse Features Table
    if not isinstance(features, Features):
        try:
            features = Features.read(features)
        except (FileNotFoundError, AttributeError):
            raise PAHFITModelError(f"No science pack found for feature table {features}.")

    # Features by kind
    fk = {v[0]['kind']: v for v in features.group_by('kind').groups}

    # feats = []
    # validity = []
    # params = []
    # bounds_low = []; bounds_high = []
    # feature_count = {}
    
    # *** Lines
    lw = fk['line']['wavelength']  # rest-frame

    inside = None
    for seg, rec in waves.items():
        ins1 = within_segment(lw * (1. + rec['z']), seg,
                              fwhm_near=validity_gaussian_fwhms,
                              wave_bounds=rec['bounds'])
        if inside is None:
            new = inside = ins1
        else:
            new = ins1 & ~inside
            inside |= ins1
            ghosts = ins1 and ~new  # line already appeared somewhere
        


    # HOW??? Helper method to add a parameter "set"?  Need to determine
    # fixed vs. bounded
    #
    # Use features' row.meta to save a mapping like param_inds:
    # {'wavelength': (type index), ...} to indicate the location of
    # parameters in the relevant I/X/T list.  On fit completion, the
    # I-list parameters can be updated with the final I-list, by
    # looping through the rows and copying the I values out into their
    # respective columns of the feature.  Along with the covariance
    # matrix, param_info is saved in features.meta, along with some
    # run details.  It can be used to quick-run a fit possibly on
    # different (but compatible) spectra, by simply updating its
    # y-list, I-list, and bounds directly and skipping the complicated
    # translation step.  Everything else is unchanged.  But this
    # should be "opt-in".  To save param_info in meta too?  Probably.
    # It will take some memory.
    

    bnds = np.array([x.bounds for x in self.waves.values()])

    # loop through all lines, find the segments of overlap, pick
    # the "best/most" overlapping segment as primary, add the
    # primary line feature, then add another power-tied "ghost"
    # feature for all other segments it "touches".  Note: lines with 

    low = bounded_min(lw)
    high = np.where(lw[:,2],lw[:,2],lw[:,0]) # handles upper bounds, if any
    

    
    # *** Starlight
    T = feat['starlight']['temperature']

    
    
    
    
    for f in features:
        count[f["kind"]] += 1

        # Validity Ranges (lines only)
        if f["kind"] == "line":
            rest_wav = f['wavelength'][0]
            v = self._line_feature_validity(f["wavelength"], f["fwhm"])
                                      
            nv = len(v)//2
            if f["kind"] == "line": # Fork the line into multiple copies
                for (ins,rng) in v:
                    fwhm = fwhm_func(
                    
                

                
            else:
                validity.append(_valid)
                                     
                            
    

    # Pre-compute counts
    

    n_feat = len(features)

    # Validity Ranges: line + dust_features only





    # TBD: Constraints
    # constraints = features.meta.get('constraints')

    if constraints:
        ties = []
        tie_groups = []











    #features_need_validity = features[[x in ('line','dust_feature') for x in t['kind']]]

    pm['feature_count'] = {}


    sl = features[features['kind'] == 'starlight']
    pm['feature_count']['starlight'] = len(sl)
    if len(sl) > 0:

    # Dust Continuum
    dc = features[features['kind'] == 'dust_continuum']
    pm['feature_count']['dust_continuum'] = len(dc)
    if len(dc) > 0:
        pass

    # Line features (compute FWHM, duplicate features as needed)
    lf = features[features['kind'] == 'line']
    pm['feature_count']['line'] = len(lf)
    for line in lf:


    # Dust Features
    df = features[features['kind'] == 'dust_feature']
    pm['feature_count']['dust_features'] = len(df)
    if len(df) > 0:
        pass

    # Attenuation
    at = features[features['kind'] == 'attenuation']

    if len(at) > 0:
        if not (at['tau'].is_fixed()[0] and at['tau'][0,0] == 0.0):
            pm['feature_count']['attenuation'] = 1 # can only be one

    # Absorption (if any)
    ab = features[features['kind'] == 'absorption']
    if len(ab) > 0:
        pm['feature_count']['absorption'] = len(ab)


