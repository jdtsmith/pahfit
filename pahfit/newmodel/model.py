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

# N.B. The flexibility in the PAHFIT model means the external and
# internal representations of features and their parameters can grow
# quite complicated.  A few tips and practical pointers:

#   - the .yaml science pack is a read-only on-disk file format, and
#     is converted directly to a Features table (descendent of
#     astropy.table.Table).  The YAML data are "hand-written"
#     (including by the advanced PAHFIT user).

#   - Features table can be edited on input and are updated on output
#     with fit results.  They can be created from yaml science pack
#     input, or saved to and read in from disk (e.g. ecsv format).

#   - There are *two* additional model-internal formats for features:
#      1. Model._features, a ModelSpec object.  This is a
#         flexible data structure that can be easily grown and edited
#         in place.  It is stored between fits.  It serves as an
#         intermediate representation between a Features table and the
#         following more efficient internal format.  It is updated
#         only when new spectra are ingested for fitting with an
#         existing model.  Otherwise it is re-built for each new Model
#         object.
#      2. Model._params, a PAHFITParams object. This is a highly
#         efficient (numpy structured array-based) format, which is
#         initialized from the ModelSpec (which knows how to translate
#         itself).  This can be thought of as operating at an
#         equivalent layers as astropy's Modeling.CompoundModel
#         machinery (though much more efficiently).

#   - On output from guess or fit, the Features table is updated from
#     the independent variable (with the help of ModelSpec, to make
#     the mapping).

#   - Constant-profile features has only a SINGLE parameter, no matter
#     the feature's kind: power or tau.

#   - A "ghost" feature is a power-tied feature (e.g. line) which
#     differs only in its FWHM; all other parameters are mapped back to
#     the main feature.  Each such ghost should have a validity range
#     within the segment for which it was created.

#   - Features can have validity over the full (combined) spectrum, or
#     over one or more validity [start, stop) ranges.  A feature with
#     more than one validity range is possible (but probably rare).
#     An example might be a broad feature unaffected by the local
#     FWHM, which nevertheless has finite coverage.  


class Model:
    """XXX Get Docs from orig model.py"""
    FLOAT_TYPE = np.float64

    def __init__(self, features: Features,
                 spectra=None, instruments=None, redshift=None):
        """Create a new PAHFIT Model.

        Parameters
        ----------

        features : `Features` or str
            Either a Features table, or a string specifying the name
            of one of the default science pack tables distributed with
            PAHFIT.

            The Features table contains information on all (groups of)
            features, their kinds, parameters (including both value and
            bounds), as well as any ties between parameters or parameter
            groups (in its meta-data).  See the documentation for
            information on the structure of this table.

        spectra : `~specutils.Spectrum1D` or iterable, optional
            A single instance or iterable of Spectrum1D spectral
            segments to fit, potentially wavelength-overlapping.  The
            canonical "instrument" name or names applicable for each
            spectrum should be encoded in each spectral segment's
            `meta' table dict under the key "instrument".  Redshift
            can be set either in the Spectrum1D object itself, or, if
            not set, specified globally with a `redshift` argument;
            see below.  Note: if `spectra` is omitted, `instruments`
            must be passed.

        instruments : str or iterable, optional
            Instead of setting the "instrument" meta-data, a string or
            iterable of (iterables of) strings can be passed, and will be used
            in place of the metadata.

        redshift : float, optional
            The unitless redshift (``z = delta(lam)/lam_0``).  Assumed
            to be 0.0 if omitted.  The passed redshift should be
            accurate to within a small fraction of the finest spectral
            resolution element in `spectra`.  Note that any redshift
            provided within the meta table of individual spectral
            segments overrides the redshift (if any) passed via this
            argument.

        See Also
        --------

        instrument : PAHFIT instrument pack definition.
        """

        # Parse Features Table
        if not isinstance(features, Features):
            try:
                features = Features.read(features)
            except (FileNotFoundError, AttributeError):
                raise PAHFITModelError(f"No science pack found for feature table {features}.")
        self.features = features

        # Process spectra or instruments
        if spectra:
            self.ingest_spectra(spectra, instruments=instruments, redshift=redshift)
        elif instruments:
            self.segments = {x: None for x in instruments}  # empty segments, for now
        else:
            raise PAHFITModelError("Model requires either spectra or instruments to create.")
        self.redshift = redshift or 0.0

    def ingest_spectra(self, spectra, instruments=None, redshift=None):
        """Prepare a set of spectra for fitting.

        Parameters
        ----------

        spectra : `~specutils.Spectrum1D` spectrum or iterable
            Spectrum or iterable of spectral segments to fit.

        instruments : str or iterable, optional
            A (list of) instrument names, each of which can be a
            string, or an iterable of strings for stitched spectra.
            This list is only used if `spectra` does not contain
            `meta['instrument']`, which is preferred.  Its length must
            correspond to the number of spectral segments passed.

        redshift : float, optional, default: 0.0
            The unitless redshift (``z = delta(lam)/lam_0``).  Assumed to be
            0.0 if omitted.  The passed redshift should be accurate to
            within a small fraction of the finest spectral resolution
            element in the passed collection of spectra.  Note that any
            redshift provided in individual spectra overrides the redshift
            (if any) passed via this argument.
        """

        # XXX Check spec_samples and self.instruments and skip most of this if unchanged.
        if not isinstance(spectra, (tuple, list)):
            spectra = (spectra,)

        self.spec_samples = sum(len(s.spectral_axis) for s in spectra)

        # Flux and Wavelength
        self.flux = np.empty(self.spec_samples, dtype=self.FLOAT_TYPE)
        self.flux_unc = np.empty_like(self.flux)
        self.wavelength = np.empty_like(self.flux)
        self.segments = {}
        off = 0
        for i, sp in enumerate(spectra):
            w = sp.spectral_axis.micron

            try:
                ins = sp.meta['instrument']  # can be a list of strings or glob
            except KeyError:
                try:
                    ins = instruments[i]  # type: ignore
                except (IndexError, TypeError):
                    raise PAHFITModelError(f"Instrument missing from input spectrum #{i}.")
            n = len(w)

            # Check compatibility between spectral wave-bounds and instrument
            bounds = [w[0], w[-1]]  # Assume monotonic, either increasing or decreasing
            if bounds[0] > bounds[1]:
                bounds = bounds.reverse()
            check_range(bounds, ins)

            z = sp.spectral_axis.redshift.value if redshift is None else redshift

            if ins in self.segments:
                raise PAHFITModelError(f"Instrument present in more than one input spectrum: {ins}")

            self.segments[ins] = dict(obs_wave=w, start=off, z=z, bounds=bounds)
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


