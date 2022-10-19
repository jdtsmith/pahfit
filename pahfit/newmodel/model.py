"""
PAHFIT optimized model object function.
"""

#from typing import ParamSpecKwargs
import numpy as np
from const import geometry
from .params import PAHFITParams
from .components import modified_blackbody, blackbody, drude, gaussian
from .pfnumba import pahfit_jit

@pahfit_jit
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
        # Build the attenuation curve by hand
        tau97 = param_map.retrieve_param(params)
        param_map.atten_curve[:] = tau97 * param_map.atten_curve
        param_map.next_feature()

        # Absorption: add any discrete absorption features with their own parameters
        for _ in range(param_map.feature_count.absorption):
            param_map.accumulate(params, drude, 3, attenuation=True)

        if param_map.atten_geom == geometry.mixed:
            param_map.y *= (1.0 - np.exp(-param_map.atten_curve)) / param_map.atten_curve
        else:  # screen
            param_map.y *= np.exp(-param_map.atten_curve)

    return param_map.y


class Model:
    """XXX Get Docs from orig model.py"""

    def __init__(self, features):
        """
        Parameters
        ----------

        features : :class:`pahfit.features.Features` object or str
            Either a Features table, or a string specifying one of the
            default science pack tables to read in as a Features table.

            The Features table contains information on all (groups of)
            features, their kinds, parameters (including both value and
            bounds), as well as any ties between parameters or parameter
            groups (in its meta-data).  See the documentation for
            information on the structure of this table.
        """

        pass


    def build_params(self, features, spectra, fwhm_func, redshift=None):
        """Create and store a fully populated :class:PAHFITParams
        parameter map.

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
                ins = sp.meta['instrument']  # can be a list of strings or glob
            except KeyError:
                raise PAHFITModelError(f"Instrument missing from input spectrum #{i}.")
            n = len(w)
            bnds = [w[0], w[-1]]  # Assume monotonic, either increasing or decreasing
            if bnds[0] > bnds[1]:
                bnds = bnds.reverse()

            check_range(bnds, ins)  # Check for compatibility between passed wave-range and instrument
            z = sp.spectral_axis.redshift.value if redshift is None else redshift
            if ins in waves:
                raise PAHFITModelError(f"Instrument present in more than one input spectrum: {ins}")

            waves[ins] = dict(obs_wave=w, start=off, z=z, bounds=bnds)
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

        # Group features in table by kind
        fk = {v[0]['kind']: v for v in features.group_by('kind').groups}

        # Internal Features data structure: a list of tuples of
        # dictionaries, each of which corresponds to one feature, like:
        #   params: {
        feat_internal = [] dict(features={})

        # COUNT all the features
        lw = fk['line']['wavelength']  # lines are special
        inside = None                  # they can be excluded
        ghosts = np.zeros_like(lw, dtype=int)  # or have ghosts

        for line in fk['line']:
            for seg, rec in waves.items():
                if within_segment(line['wavelength'] * (1. + rec['z']), seg,
                                  fwhm_near=validity_gaussian_fwhms,
                                  wave_bounds=rec['bounds']):
            if inside is None:
                inside = ins_new
            else:
                ghosts[ins_new & inside] += 1  # line was already in
                inside |= ins_new
        if inside is None:
            n_features = len(features)
        else:
            n_features = len(features) - np.count_nonzero(~inside) + ghosts.sum()

        kept_lines = fk['line'][inside]




        n_indep = 0





        # pp = PAHFITParams(). # XXX maybe doesn't need to be separate. This will cache the jitclass!

        # TODO: Check cached param_map if passed.  What checks to be done?
        # Just opt-in and let the user be responsible.


