"""Parameter structure for PAHFIT model fit"""

# https://numba.discourse.group/t/best-practices-for-complex-structured-input/1406

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
    from numba import i4, f8, from_dtype
    feat_spec = dict(starlight=i4, dust_continuum=i4,n_mbb=i4,line=i4,
                     dust_features=i4,attenuation=i4, absorption=i4)
    param_spec = dict(y=f8[:], wavelength=f8[:],
                      bounds_low=i4[:], bounds_high=i4[:],
                      features=from_dtype(feature_dtype)[:],
                      validity=i4[:],
                      params=from_dtype(params_dtype)[:],
                      fixed=i4[:],
                      tied=from_dtype(tied_dtype)[:],
                      tie_groups=i4[:],
                      const_profile=f8[:],
                      atten_geom = i4,
                      atten_curve = f8[:])
else:
    feat_spec, param_spec = (None, None)

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

        # Parameter lists:
        self.features = np.empty(n_feat, dtype=feature_dtype) # F-list
        self.validity = np.empty(n_val, dtype=np.int32)       # V-list
        self.params = np.empty(n_param, dtype=params_dtype)   # P-list
        self.fixed = np.empty(n_fixed, dtype=np.int32)        # X-list

        # Constraintes
        if n_tied>0:
            self.tied = np.empty(n_param, dtype=tied_dtype)   # T-list
            self.tie_groups = np.empty(n_tie_groups, dtype=np.int32) # G-list

        # Constant Profile features
        if n_cp>0:
            self.const_profile = np.empty(n_cp, dtype=np.float64) # C-list

        # Attenuation
        self.atten_geom = atten_geom
        self.atten_curve = np.empty(m, dtype=np.float64) # A-list

    #def update(self, bounds_low, bounds_high):



class PAHFITParamsWrapper:
    def __init__(self,features, spectra, fwhm_func, redshift = None):
        """Create and return the internal parameter map, which provides
        all the auxiliary parameter and constraint information needed for
        efficient yet flexible calculation of the PAHFIT model function.

        Arguments:
        ----------

        features: A `pahfit.features.Features` table, which contains
          information on all (groups of) features, their kinds, parameters
          (including both value and bounds), as well as any ties between
          parameters or parameter groups.  See the documentation for
          information on the structure of this table, which is an internal
          implementation detail.

        spectra: A single instance or an iterable of (potentially
          wavelength-overlapping) specutils.Spectrum1D objects to fit.
          The canonical "instrument" name for each segment must be encoded
          in its meta table dict under the key "instrument".  Redshift can
          also be set in the meta table (see also redshift, below).

        fwhm_func: A function of two arguments, which provides FWHM information
          from the instrument pack:

             fwhm = fwhm_func(instrument, observed_frame_wavelength_microns)

          which returns the fwhm (in microns) based on segment instrument
          name and (observed frame) wavelength of the line, in microns.

          Note: Normally, line FWHM is considered a detail of the
          instrument pack, and never needs consideration.  If a line's
          FWHM is provided in advance in the features table (with or
          without bounds), it will OVERRIDE the value in the instrument
          pack.  In that case, it is imperative that only a single
          spectral segment is being fit, or that the resolution is
          otherwise the same at the line location.  Upon fit completion,
          line FWHM will ONLY be updated for output in the features table
          for any lines whose FWHM was provided in advance (and not
          fixed).

        redshift: The unitless redshift (z = delta(lam)/lam_0).  Assumed
          to be 0.0 if omitted.  The passed redshift should be accurate to
          within a small fraction of the finest spectral resolution
          element in the passed collection of spectra.  Note that any
          passed redshift overrides the redshift set in individual
          spectra.

        Returns:
        --------

        A populated pahfit.model.param.PAHFITParams object, which includes
        scalars and (structured) numpy arrays.  This structure is an
        internal implementation detail of the PAHFIT model, and not for
        external use.  In all cases the arrays are simple 1D arrays,
        either integer, double floating point, or named structured arrays
        of integers/doubles.  See the documentation for details on the
        various param_map arrays.

        Note: it is possible to re-use a param_map object between fits, assuming
        none of the model details have changed.
        """

        #pp = PAHFITParams()

        # TODO: Check cached param_map if passed.  What checks?


        # *** Ingest Spectra
        if not isinstance(spectra, (tuple, list)):
            self.spectra = (spectra,)
        else:
            self.spectra = spectra
        spec_samples = sum(len(s.spectral_axis) for s in self.spectra)

        flux = np.empty(spec_samples, dtype=np.float64)
        flux_unc = np.empty(spec_samples, dtype=np.float64)
        wavelength = np.empty(spec_samples, dtype=np.float64)

        # Flux and Wavelength
        off = 0
        self.waves = {}
        for i, sp in enumerate(self.spectra):
            w = sp.spectral_axis.micron

            try:
                ins = sp.meta['instrument']
            except KeyError:
                raise PAHFITModelError(f"Instrument missing from input spectrum #{i}.")
            n = len(w)
            bd = [w[0], w[-1]] # Assume monotonic, either increasing or decreasing
            z = sp.spectral_axis.redshift.value if redshift is None else redshift
            self.waves[ins]=dict(obs_wave = w, start = off, z = z, 
                                 bounds = bd.reverse() if bd[0] > bd[1] else bd)

            f = sp.flux
            f_unc = sp.uncertainty
            if f_unc is None:
                raise PAHFITModelError(f"Flux uncertainty missing: {ins}") 

            # De-redshift
            if not z == 0.0:
                w /= (1. + z)
                f *= (1. + z)  # Conserves power
                f_unc *= (1. + z)

            wavelength[off:off + n] = w
            flux[off:off + n] = f
            flux_unc[off:off + n] = f_unc

            off += n

        fk = {v[0]['kind']: v for v in features.group_by('kind').groups}

        # TBD: Constraints
        # constraints = features.meta.get('constraints')

        # ** Features
        feats = []
        validity = []
        params = []
        bounds_low = []; bounds_high = []
        count = {}

        # Lines
        lw = fk['line']['wavelength']
        
        
        
        
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









        if constraints:
            ties = []
            tie_groups = []






        # Starlight
        T = feat['starlight']['temperature']





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


