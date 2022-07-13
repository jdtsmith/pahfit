"""
PAHFIT optimized model function
"""

import numpy as np

from .components import blackbody, drude, gaussian, power, amplitude

# * Parameter Setup
# XXX what data type should be returned here, for NUMBA compliance?  See
# https://numba.discourse.group/t/best-practices-for-complex-structured-input/1406


def _create_param_map(features, spectra, fwhm_func, redshift = None):
    """Create and return the internal parameter map, which provides
    all the auxiliary parameter and constraint information needed for
    efficient calculation of the PAHFIT model function.

    Arguments:
    ----------

    features: A `pahfit.features.Features` table, which contains
      information on all (groups of) features, their kinds, parameters
      (with both value and bounds), as well as any ties between parameters
      or parameter groups.  See the documentation for information on
      the structure of this table.

    spectra: A single instance or an iterable of (potentially
      wavelength-overlapping) specutils.Spectrum1D objects to fit.
      The canonical "instrument" name for each segment must be encoded
      in its meta table.  Redshift can also be set (see also redshift,
      below).

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

    redshift: The unitless redshift (z = delta(lam)/lam_0).  Assumed to be
      0.0 if omitted.  The passed redshift should be accurate to
      within a fraction of the finest spectral resolution element in
      the passed collection of spectra.  Note that any passed redshift
      overrides the redshift set in individual Spectrum1D objects
      passed in spectra.

    Returns:
    --------

    A populated pahfit.model.param.ParamMap object, which includes
    scalars and (structured) numpy arrays.  This structure is an
    internal implementation detail of the PAHFIT model, and not for
    external use.  In all cases the arrays are simple 1D arrays,
    either integer, double floating point, or named structured arrays
    of integers/doubles.  See the documentation for details on the
    various param_map arrays.

    Note: it is possible to /re-use/ a param_map object between fits, assuming
    none of the model details have changed.

    """
    pm = {}

    # TODO: Check cached param_map if passed.  What checks?
    
    if not isinstance(spectra, (tuple, list)):
        spectra = (spectra,)

    # Ingest Spectra
    spec_samples = sum(len(s.spectral_axis) for s in spectra)
        
    flux = np.empty(spec_samples, dtype=np.float64)
    flux_unc = np.empty(spec_samples, dtype=np.float64)

    l
    
    off = 0
    waves = []
    for sp in spectra:
        w = sp.spectral_axis.micron
        n = len(w)
        bd = [w[0], w[-1]] # Assume monotonic, either increasing or decreasing
        waves.append(dict(instrument = sp.meta['instrument'], obs_wave = w,
                          bounds = bd.reverse() if bd[0] > bd[1] else bd))
        
        z = sp.spectral_axis.redshift if redshift is None else redshift
        f = sp.flux
        f_unc = sp.uncertainty
        if f_unc is None:
            raise Exception("Flux uncertainty missing") # TODO: Use real error
        if not z == 0.0:
            w /= (1. + z)
            f *= (1. + z)  # Conserve power
            f_unc *= (1. + z)
            
        np.insert(pm['wavelength'], off, w)
        np.insert(flux, off, f)
        np.insert(flux_unc, off, f_unc)
        off += n

    #features_need_validity = features[[x in ('line','dust_feature') for x in t['kind']]]

    pm['feature_count'] = {}
    # Starlight
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
    
    
    

def _feature_validity(lam0, fwhm, , drude=False):
    
                      
# ** Model Function

def pahfit_function(params, param_map):

    """Calculate and return the PAHFIT model function.

    Arguments
    ---------

    PARAMS: The 1D array of independent parameters, as passed by the
      optimizer.  Note that fixed parameters are not included (see
      FIXED, below).

    PARAM_MAP: A named tuple of arrays identifying all the
      independent, fixed, and tied parameters.  This argument is
      automatically constructed by _create_param_map.  See the
      documentation for information on the internal details.
    """
    


    pass
