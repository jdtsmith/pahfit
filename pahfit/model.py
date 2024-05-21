from specutils import Spectrum1D
from astropy import units as u
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate, integrate

from pahfit import units
from pahfit.features.util import bounded_is_fixed, bounded_is_missing
from pahfit.features import Features
from pahfit import instrument
from pahfit.errors import PAHFITModelError
from pahfit.component_models import BlackBody1D, S07_attenuation
from pahfit.fitter import Fitter
from pahfit.apfitter import APFitter


class Model:
    """This class acts as the main API for PAHFIT.

    The users deal with model objects, of which the state is modified
    during initalization, initial guessing, and fitting. What the model
    STORES is a description of the physics: what features are there and
    what are their properties, regardless of the instrument with which
    those features are observed. The methods provided by this class,
    form the connection between those physics, and what is observed.
    During fitting and plotting, those physics are converted into a
    model for the observation, by applying instrumental parameters from
    the instrument.py module.

    The main thing that defines a model, is the features table, loaded
    from a YAML file given to the constructor. After construction, the
    Model can be edited by accessing the stored features table directly.
    Changing numbers in this table, is allowed, and the updated numbers
    will be reflected when the next fit or initial guess happens. At the
    end of these actions, the fit or guess results are stored in the
    same table.

    The model can be saved.

    The model can be copied.

    Attributes
    ----------
    features : Features
        Instance of the Features class. Can be edited on-the-fly.
        Non-breaking behavior by the user is expected. Changes will be
        reflected at the next fit, guess, or plot call.

    """

    def __init__(self, features: Features):
        """
        Parameters
        ----------
        features: Features
            Features table.

        """
        self.features = features

        # If features table does not originate from a previous fit, and
        # hence has no unit yet, we initialize it as an empty dict.
        if "user_unit" not in self.features.meta:
            self.features.meta["user_unit"] = {}

        # store fit_info dict of last fit
        self.fit_info = None

    @classmethod
    def from_yaml(cls, pack_file):
        """
        Generate feature table from YAML file.

        Parameters
        ----------
        pack_file : str
            Path to YAML file, or name of one of the default YAML files.

        Returns
        -------
        Model instance

        """
        features = Features.read(pack_file)
        return cls(features)

    @classmethod
    def from_saved(cls, saved_model_file):
        """
        Parameters
        ----------
        saved_model_file : str
           Path to file generated by Model.save()

        Returns
        -------
        Model instance
        """
        # features.read automatically switches to astropy table reader.
        # Maybe needs to be more advanced here in the future.
        features = Features.read(saved_model_file, format="ascii.ecsv")
        return cls(features)

    def save(self, fn, **write_kwargs):
        """Save the model to disk.

        Only ECSV supported for now. Models saved this way can be read
        back in, with metadata.

        TODO: store details about the fit results somehow. Uncertainties
        (covariance matrix) should be retrievable. Use Table metadata?

        Parameters
        ----------
        fn : file name

        **write_kwargs : kwargs passed to astropy.table.Table.write

        """
        if fn.split(".")[-1] != "ecsv":
            raise NotImplementedError("Only ascii.ecsv is supported for now")

        self.features.write(fn, format="ascii.ecsv", **write_kwargs)

    def _status_message(self):
        out = "Model features ("
        if self.fit_info is None:
            out += "not "
        out += "fitted)\n"
        return out

    def __repr__(self):
        return self._status_message() + self.features.__repr__()

    def _repr_html_(self):
        return self._status_message() + self.features._repr_html_()

    def guess(
        self,
        spec: Spectrum1D,
        redshift=None,
        integrate_line_flux=False,
        calc_line_fwhm=True,
    ):
        """Make an initial guess of the physics, based on the given
        observational data.

        Parameters
        ----------
        spec : Spectrum1D
            1D (not 2D or 3D) spectrum object, containing the
            observational data. (TODO: should support list of spectra,
            for the segment-based joint fitting). Initial guess will be
            based on the flux in this spectrum.

            spec.meta['instrument'] : str or list of str
                Qualified instrument name, see instrument.py. This will
                determine what the line widths are, when going from the
                features table to a fittable/plottable model.

        redshift : float
            Redshift used to shift from the physical model, to the
            observed model.

            If None, will be taken from spec.redshift

        integrate_line_flux : bool
            Use the trapezoid rule to estimate line fluxes. Default is
            False, where a simpler line guess is used (~ median flux).

        calc_line_fwhm : bool
            Default, True. Can be set to False to disable the instrument
            model during the guess, as to avoid overwriting any manually
            specified line widths.

        Returns
        -------
        Nothing, but internal feature table is updated.

        """
        inst, z = self._parse_instrument_and_redshift(spec, redshift)
        # save these as part of the model (will be written to disk too)
        self.features.meta["redshift"] = inst
        self.features.meta["instrument"] = z

        # parse spectral data
        self.features.meta["user_unit"]["flux"] = spec.flux.unit
        _, _, _, lam, flux, _ = self._convert_spec_data(spec, z)
        wmin = min(lam)
        wmax = max(lam)

        # simple linear interpolation function for spectrum
        sp = interpolate.interp1d(lam, flux)

        # we will repeat this loop logic several times
        def loop_over_non_fixed(kind, parameter, estimate_function, force=False):
            for row_index in np.where(self.features["kind"] == kind)[0]:
                row = self.features[row_index]
                if not bounded_is_fixed(row[parameter]) or force:
                    guess_value = estimate_function(row)
                    # print(f"{row['name']}: setting {parameter} to {guess_value}")
                    self.features[row_index][parameter][0] = guess_value

        # guess starting point of bb
        def starlight_guess(row):
            bb = BlackBody1D(1, row["temperature"][0])
            w = wmin + 0.1  # the wavelength used to compare
            if w < 5:
                # wavelength is short enough to not have numerical
                # issues. Evaluate both at w.
                amp_guess = sp(w) / bb(w)
            else:
                # wavelength too long for stellar BB. Evaluate BB at
                # 5 micron, and spectrum data at minimum wavelength.
                wsafe = 5
                amp_guess = sp(w) / bb(wsafe)

            return amp_guess

        loop_over_non_fixed("starlight", "tau", starlight_guess)

        # count number of blackbodies in the model
        nbb = len(self.features[self.features["kind"] == "dust_continuum"])

        def dust_continuum_guess(row):
            temp = row["temperature"][0]
            fmax_lam = 2898.0 / temp
            bb = BlackBody1D(1, temp)
            if fmax_lam >= wmin and fmax_lam <= wmax:
                w_ref = fmax_lam
            elif fmax_lam > wmax:
                w_ref = wmax
            else:
                w_ref = wmin

            flux_ref = np.median(flux[(lam > w_ref - 0.2) & (lam < w_ref + 0.2)])
            amp_guess = flux_ref / bb(w_ref)
            return amp_guess / nbb

        loop_over_non_fixed("dust_continuum", "tau", dust_continuum_guess)

        def line_fwhm_guess(row):
            w = row["wavelength"][0]
            if not instrument.within_segment(w, inst):
                return 0

            fwhm = instrument.fwhm(inst, w, as_bounded=True)[0][0]
            return fwhm

        def amp_guess(row, fwhm):
            w = row["wavelength"][0]
            if not instrument.within_segment(w, inst):
                return 0

            factor = 1.5
            wmin = w - factor * fwhm
            wmax = w + factor * fwhm
            lam_window = (lam > wmin) & (lam < wmax)
            xpoints = lam[lam_window]
            ypoints = flux[lam_window]
            if np.count_nonzero(lam_window) >= 2:
                # difference between flux in window and flux around it
                power_guess = integrate.trapezoid(flux[lam_window], lam[lam_window])
                # subtract continuum estimate, but make sure we don't go negative
                continuum = (ypoints[0] + ypoints[-1]) / 2 * (xpoints[-1] - xpoints[0])
                if continuum < power_guess:
                    power_guess -= continuum
            else:
                power_guess = 0
            return power_guess / fwhm

        # Same logic as in the old function: just use same amp for all
        # dust features.
        some_flux = 0.5 * np.median(flux)
        loop_over_non_fixed("dust_feature", "power", lambda row: some_flux)

        if integrate_line_flux:
            # calc line amplitude using instrumental fwhm and integral over data
            loop_over_non_fixed(
                "line", "power", lambda row: amp_guess(row, line_fwhm_guess(row))
            )
        else:
            loop_over_non_fixed("line", "power", lambda row: some_flux)

        # Set the fwhms in the features table. Slightly different logic,
        # as the fwhm for lines are masked by default. TODO: leave FWHM
        # masked for lines, and instead have a sigma_v option. Any
        # requirements to guess and fit the line width, should be
        # encapsulated in sigma_v (the "broadening" of the line), as
        # opposed to fwhm which is the normal instrumental width.
        if calc_line_fwhm:
            for row_index in np.where(self.features["kind"] == "line")[0]:
                row = self.features[row_index]
                if np.ma.is_masked(row["fwhm"]):
                    self.features[row_index]["fwhm"] = (
                        line_fwhm_guess(row),
                        np.nan,
                        np.nan,
                    )
                elif not bounded_is_fixed(row["fwhm"]):
                    self.features[row_index]["fwhm"]["val"] = line_fwhm_guess(row)

    @staticmethod
    def _convert_spec_data(spec, z):
        """Convert Spectrum1D Quantities to fittable numbers.

        The unit of the input spectrum has to be a multiple of MJy / sr,
        the internal intensity unit. The output of this function
        consists of simple unitless arrays (the numbers in these arrays
        are assumed to be consistent with the internal units).

        Also corrects for redshift.

        Returns
        -------
        x, y, unc: wavelength in micron, flux, uncertainty

        lam, flux, unc: wavelength in micron, flux, uncertainty
            corrected for redshift

        """
        if not spec.flux.unit.is_equivalent(units.intensity):
            raise PAHFITModelError(
                "For now, PAHFIT only supports intensity units, i.e. convertible to MJy / sr."
            )
        flux_obs = spec.flux.to(units.intensity).value
        lam_obs = spec.spectral_axis.to(u.micron).value
        unc_obs = (spec.uncertainty.array * spec.flux.unit).to(units.intensity).value

        # transform observed wavelength to "physical" wavelength
        lam = lam_obs / (1 + z)  # wavelength shorter
        flux = flux_obs * (1 + z)  # energy higher
        unc = unc_obs * (1 + z)  # uncertainty scales with flux
        return lam_obs, flux_obs, unc_obs, lam, flux, unc

    def fit(
        self,
        spec: Spectrum1D,
        redshift=None,
        maxiter=1000,
        verbose=True,
        use_instrument_fwhm=True,
    ):
        """Fit the observed data.

        The model setup is based on the features table and instrument
        specification.

        The last fit results can accessed through the variable
        model.astropy_result. The results are also stored back to the
        model.features table.

        CAVEAT: any features that do not overlap with the data range
        will not be included in the model, for performance and numerical
        stability. Their values in the features table will be left
        untouched.

        Parameters
        ----------
        spec : Spectrum1D
            1D (not 2D or 3D) spectrum object, containing the
            observational data. (TODO: should support list of spectra,
            for the segment-based joint fitting). Initial guess will be
            based on the flux in this spectrum.

            spec.meta['instrument'] : str or list of str
                Qualified instrument name, see instrument.py. This will
                determine what the line widths are, when going from the
                features table to a fittable/plottable model.

        redshift : float
            Redshift used to shift from the physical model, to the
            observed model.

            If None, will be taken from spec.redshift

        maxiter : int
            maximum number of fitting iterations

        verbose : boolean
            set to provide screen output

        use_instrument_fwhm : bool
            Use the instrument model to calculate the fwhm of the
            emission lines, instead of fitting them, which is the
            default behavior. This can be set to False to set the fwhm
            manually using the value in the science pack. If False and
            bounds are provided on the fwhm for a line, the fwhm for
            this line will be fit to the data.

        """
        # parse spectral data
        self.features.meta["user_unit"]["flux"] = spec.flux.unit
        inst, z = self._parse_instrument_and_redshift(spec, redshift)
        x, _, _, lam, flux, unc = self._convert_spec_data(spec, z)

        # save these as part of the model (will be written to disk too)
        self.features.meta["redshift"] = inst
        self.features.meta["instrument"] = z

        # check if observed spectrum is compatible with instrument model
        instrument.check_range([min(x), max(x)], inst)

        self._set_up_fitter(inst, z, x=x, use_instrument_fwhm=use_instrument_fwhm)
        self.fitter.fit(lam, flux, unc, maxiter=maxiter)

        # copy the fit results to the features table
        self._ingest_fit_result_to_features()

        if verbose:
            print(self.fitter.message)

    def _ingest_fit_result_to_features(self):
        """Copy the results from the Fitter to the features table

        This is a utility method, executed only at the end of fit(),
        where Fitter.fit() has been applied.

        """
        # iterate over the list stored in fitter, so we only get
        # components that were set up by _set_up_fitter. Having an
        # ENABLED/DISABLED flag for every feature would be a nice
        # alternative (and clear for the user).

        self.features.meta["fitter_message"] = self.fitter.message

        for name in self.fitter.components():
            for column, value in self.fitter.get_result(name).items():
                try:
                    i = np.where(self.features["name"] == name)[0]
                    # deal with fwhm usually being masked
                    if not bounded_is_missing(self.features[column][i]):
                        self.features[column]["val"][i] = value
                    else:
                        self.features[column][i] = (value, np.nan, np.nan)
                except Exception as e:
                    print(
                        f"Could not assign to name {name} in features table. Some diagnostic output below"
                    )
                    print(f"Index i is {i}")
                    print("Features table:", self.features)
                    raise e

    def plot(
        self,
        spec=None,
        redshift=None,
        use_instrument_fwhm=False,
        label_lines=False,
        scalefac_resid=2,
        **errorbar_kwargs,
    ):
        """Plot model, and optionally compare to observational data.

        Parameters
        ----------
        spec : Spectrum1D
            Observational data. The units should be compatible with the
            data that were used for the fit, but it does not have to be
            the exact same spectrum. The spectrum will be converted to
            internal units before plotting.

        redshift : float
            Redshift used to shift from the physical model, to the
            observed model. If None, it will be taken from spec.redshift

        use_instrument_fwhm : bool
            For the lines, the default is to use the fwhm values
            contained in the Features table. When set to True, the fwhm
            will be determined by the instrument model instead.

        label_lines : bool
            Add labels with the names of the lines, at the position of
            each line.

        scalefac_resid : float
            Factor multiplying the standard deviation of the residuals
            to adjust plot limits.

        errorbar_kwargs : dict
            Customize the data points plot by passing the given keyword
            arguments to matplotlib.pyplot.errorbar.

        """
        inst, z = self._parse_instrument_and_redshift(spec, redshift)
        _, _, _, lam, flux, unc = self._convert_spec_data(spec, z)
        enough_samples = max(10000, len(spec.wavelength))
        lam_mod = np.logspace(np.log10(min(lam)), np.log10(max(lam)), enough_samples)

        fig, axs = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # spectrum and best fit model
        ax = axs[0]
        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.minorticks_on()
        ax.tick_params(
            axis="both", which="major", top="on", right="on", direction="in", length=10
        )
        ax.tick_params(
            axis="both", which="minor", top="on", right="on", direction="in", length=5
        )

        ext_model = None
        has_att = "attenuation" in self.features["kind"]
        has_abs = "absorption" in self.features["kind"]
        if has_att:
            row = self.features[self.features["kind"] == "attenuation"][0]
            tau = row["tau"][0]
            ext_model = S07_attenuation(tau_sil=tau)(lam_mod)

        if has_abs:
            raise NotImplementedError(
                "plotting absorption features not implemented yet"
            )

        if has_att or has_abs:
            ax_att = ax.twinx()  # axis for plotting the extinction curve
            ax_att.tick_params(which="minor", direction="in", length=5)
            ax_att.tick_params(which="major", direction="in", length=10)
            ax_att.minorticks_on()
            ax_att.plot(lam_mod, ext_model, "k--", alpha=0.5)
            ax_att.set_ylabel("Attenuation")
            ax_att.set_ylim(0, 1.1)
        else:
            ext_model = np.ones(len(lam_mod))

        # Define legend lines
        Leg_lines = [
            mpl.lines.Line2D([0], [0], color="k", linestyle="--", lw=2),
            mpl.lines.Line2D([0], [0], color="#FE6100", lw=2),
            mpl.lines.Line2D([0], [0], color="#648FFF", lw=2, alpha=0.5),
            mpl.lines.Line2D([0], [0], color="#DC267F", lw=2, alpha=0.5),
            mpl.lines.Line2D([0], [0], color="#785EF0", lw=2, alpha=1),
            mpl.lines.Line2D([0], [0], color="#FFB000", lw=2, alpha=0.5),
        ]

        # local utility
        def tabulate_components(kind):
            ss = {}
            for name in self.features[self.features["kind"] == kind]["name"]:
                ss[name] = self.tabulate(
                    inst, z, lam_mod, self.features["name"] == name
                )
            return {name: s.flux.value for name, s in ss.items()}

        cont_y = np.zeros(len(lam_mod))
        if "dust_continuum" in self.features["kind"]:
            # one plot for every component
            for y in tabulate_components("dust_continuum").values():
                ax.plot(lam_mod, y * ext_model, "#FFB000", alpha=0.5)
                # keep track of total continuum
                cont_y += y

        if "starlight" in self.features["kind"]:
            star_y = self.tabulate(
                inst, z, lam_mod, self.features["kind"] == "starlight"
            ).flux.value
            ax.plot(lam_mod, star_y * ext_model, "#ffB000", alpha=0.5)
            cont_y += star_y

        # total continuum
        ax.plot(lam_mod, cont_y * ext_model, "#785EF0", alpha=1)

        # now plot the dust bands and lines
        if "dust_feature" in self.features["kind"]:
            for y in tabulate_components("dust_feature").values():
                ax.plot(
                    lam_mod,
                    (cont_y + y) * ext_model,
                    "#648FFF",
                    alpha=0.5,
                )

        if "line" in self.features["kind"]:
            for name, y in tabulate_components("line").items():
                ax.plot(
                    lam_mod,
                    (cont_y + y) * ext_model,
                    "#DC267F",
                    alpha=0.5,
                )
                if label_lines:
                    i = np.argmax(y)
                    # ignore out of range lines
                    if i > 0 and i < len(y) - 1:
                        w = lam_mod[i]
                        ax.text(
                            w,
                            y[i],
                            name,
                            va="center",
                            ha="center",
                            rotation="vertical",
                            bbox=dict(facecolor="white", alpha=0.75, pad=0),
                        )

        ax.plot(lam_mod, self.tabulate(inst, z, lam_mod).flux.value, "#FE6100", alpha=1)

        # data
        default_kwargs = dict(
            fmt="o",
            markeredgecolor="k",
            markerfacecolor="none",
            ecolor="k",
            elinewidth=0.2,
            capsize=0.5,
            markersize=6,
        )

        ax.errorbar(lam, flux, yerr=unc, **(default_kwargs | errorbar_kwargs))

        ax.set_ylim(0)
        ax.set_ylabel(r"$\nu F_{\nu}$")

        ax.legend(
            Leg_lines,
            [
                "S07_attenuation",
                "Spectrum Fit",
                "Dust Features",
                r"Atomic and $H_2$ Lines",
                "Total Continuum Emissions",
                "Continuum Components",
            ],
            prop={"size": 10},
            loc="best",
            facecolor="white",
            framealpha=1,
            ncol=3,
        )

        # residuals = data in rest frame - (model evaluated at rest frame wavelengths)
        res = flux - self.tabulate(inst, 0, lam).flux.value
        std = np.nanstd(res)
        ax = axs[1]

        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.tick_params(
            axis="both", which="major", top="on", right="on", direction="in", length=10
        )
        ax.tick_params(
            axis="both", which="minor", top="on", right="on", direction="in", length=5
        )
        ax.minorticks_on()

        # Custom X axis ticks
        ax.xaxis.set_ticks(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30, 40]
        )

        ax.axhline(0, linestyle="--", color="gray", zorder=0)
        ax.plot(
            lam,
            res,
            "ko",
            fillstyle="none",
            zorder=1,
            markersize=errorbar_kwargs.get("markersize", None),
            alpha=errorbar_kwargs.get("alpha", None),
            linestyle="none",
        )
        ax.set_ylim(-scalefac_resid * std, scalefac_resid * std)
        ax.set_xlim(np.floor(np.amin(lam)), np.ceil(np.amax(lam)))
        ax.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax.set_ylabel("Residuals [%]")

        # scalar x-axis marks
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        fig.subplots_adjust(hspace=0)
        return fig

    def copy(self):
        """Copy the model.

        Main use case: use this model as a parent model for more
        fits.

        Currently uses copy.deepcopy. We should do something smarter if
        we run into memory problems or sluggishness.

        Returns
        -------
        model_copy : Model
        """
        # A standard deepcopy works fine!
        return copy.deepcopy(self)

    def tabulate(
        self,
        instrumentname,
        redshift=0,
        wavelengths=None,
        feature_mask=None,
    ):
        """Tabulate model flux on a wavelength grid, and export as Spectrum1D

        The flux unit will be the same as the last fitted spectrum, or
        dimensionless if the model is tabulated before being fit.

        Parameters
        ----------
        wavelengths : Spectrum1D or array-like
            Wavelengths in micron in the observed frame. Will be
            multiplied with 1/(1+z) if redshift z is given, so that the
            model is evaluated in the rest frame as intended. If a
            Spectrum1D is given, wavelengths.spectral_axis will be
            converted to micron and then used as wavelengths.

        instrumentname : str or list of str
            Qualified instrument name, see instrument.py. This
            determines the wavelength range of features to be included.
            The FWHM of the unresolved lines will be determined by the
            value in the features table, instead of the instrument. This
            allows us to visualize the fitted line widths in the
            spectral overlap regions.

        redshift : float
            The redshift is needed to evaluate the flux model at the
            right rest wavelengths.

        feature_mask : array of bool of length len(features)
            Mask used to select specific rows of the feature table. In
            most use cases, this mask can be made by applying a boolean
            operation to a column of self.features, e.g.
            model.features['wavelength'] > 8.5

        Returns
        -------
        model_spectrum : Spectrum1D
            The flux model, evaluated at the given wavelengths, packaged
            as a Spectrum1D object.
        """
        z = 0 if redshift is None else redshift

        # decide which wavelength grid to use
        if wavelengths is None:
            ranges = instrument.wave_range(instrumentname)
            if isinstance(ranges[0], float):
                wmin, wmax = ranges
            else:
                # In case of multiple ranges (multiple segments), choose
                # the min and max instead
                wmin = min(r[0] for r in ranges)
                wmax = max(r[1] for r in ranges)
            wfwhm = instrument.fwhm(instrumentname, wmin, as_bounded=True)[0, 0]
            wav = np.arange(wmin, wmax, wfwhm / 2) * u.micron
        elif isinstance(wavelengths, Spectrum1D):
            wav = wavelengths.spectral_axis.to(u.micron)
        else:
            # any other iterable will be accepted and converted to array
            wav = np.asarray(wavelengths) * u.micron

        # apply feature mask, make sub model, and set up functional
        if feature_mask is not None:
            features_copy = self.features.copy()
            features_to_use = features_copy[feature_mask]
        else:
            features_to_use = self.features

        # if nothing is in range, return early with zeros
        if len(features_to_use) == 0:
            return Spectrum1D(
                spectral_axis=wav, flux=np.zeros(wav.shape) * u.dimensionless_unscaled
            )

        alt_model = Model(features_to_use)

        # Always use the current FWHM here (use_instrument_fwhm would
        # overwrite the value in the instrument overlap regions!)

        # need to wrap in try block to avoid bug: if all components are
        # removed (because of wavelength range considerations), it won't work
        try:
            fitter = alt_model._set_up_fitter(
                instrumentname, z, use_instrument_fwhm=False
            )
        except PAHFITModelError:
            return Spectrum1D(
                spectral_axis=wav, flux=np.zeros(wav.shape) * u.dimensionless_unscaled
            )

        # shift the "observed wavelength grid" to "physical wavelength grid
        wav /= 1 + z
        flux_values = fitter.evaluate_model(wav.value)

        # apply unit stored in features table (comes from from last fit
        # or from loading previous result from disk)
        if "flux" not in self.features.meta["user_unit"]:
            flux_quantity = flux_values * u.dimensionless_unscaled
        else:
            user_unit = self.features.meta["user_unit"]["flux"]
            flux_quantity = (flux_values * units.intensity).to(user_unit)

        return Spectrum1D(spectral_axis=wav, flux=flux_quantity)

    def _excluded_features(self, instrumentname, redshift, lam_obs=None):
        """Determine excluded features Based on instrument wavelength range.

         instrumentname : str
            Qualified instrument name

         lam_obs : array
            Optional observed wavelength grid. Exclude any lines and
            dust features outside of this range.

        Returns
        -------
        array of bool, same length as self.features, True where features
        are far outside the wavelength range.
        """
        lam_feature_obs = self.features["wavelength"]["val"] * (1 + redshift)

        # has wavelength and not within instrument range
        is_outside = ~instrument.within_segment(lam_feature_obs, instrumentname)

        # also apply observed range if provided
        if lam_obs is not None:
            is_outside |= (lam_feature_obs < np.amin(lam_obs)) | (
                lam_feature_obs > np.amax(lam_obs)
            )

        # restriction on the kind of feature that can be excluded
        excludable = ["line", "dust_feature", "absorption"]
        is_excludable = np.logical_or.reduce(
            [kind == self.features["kind"] for kind in excludable]
        )

        return is_outside & is_excludable

    def _set_up_fitter(
        self, instrumentname, redshift, x=None, use_instrument_fwhm=True
    ):
        """Convert features table to Fitter instance, set self.fitter.

        For every row of the features table, calls a function of Fitter
        API to register an appropriate component. Finalizes the Fitter
        at the end (details of this step depend on the Fitter subclass).

        Any unit conversions and model-specific things need to happen
        BEFORE giving them to the fitters.
        - The instrument-derived FWHM is determined here using the
          instrument model (the Fitter does not need to know about this
          detail).
        - Features outside the appropriate wavelength range should not
          be added to the Fitter: the "trimming" is done here, using the
          given wavelength range lam (optional).

        TODO: flags to indicate which features were excluded.

        """
        # Fitting implementation can be changed by choosing another
        # Fitter class. TODO: make this configurable.
        self.fitter = APFitter()

        excluded = self._excluded_features(instrumentname, redshift, x)

        def cleaned(features_tuple3):
            val = features_tuple3[0]
            if bounded_is_fixed(features_tuple3):
                return val
            else:
                vmin = -np.inf if np.isnan(features_tuple3[1]) else features_tuple3[1]
                vmax = np.inf if np.isnan(features_tuple3[2]) else features_tuple3[2]
                return np.array([val, vmin, vmax])

        for row in self.features[~excluded]:
            kind = row["kind"]
            name = row["name"]

            if kind == "starlight":
                self.fitter.add_feature_starlight(
                    name, cleaned(row["temperature"]), cleaned(row["tau"])
                )

            elif kind == "dust_continuum":
                self.fitter.add_feature_dust_continuum(
                    name, cleaned(row["temperature"]), cleaned(row["tau"])
                )

            elif kind == "line":
                if use_instrument_fwhm:
                    # one caveat here: redshift. Correct way to do it:
                    # 1. shift to observed wav; 2. evaluate fwhm at
                    # oberved wav; 3. shift back to rest frame wav
                    # (width in rest frame will be narrower than
                    # observed width)
                    w_obs = row["wavelength"]["val"] * (1.0 + redshift)
                    # returned value is tuple (value, min, max). And
                    # min/max are already masked in case of fixed value
                    # (output of instrument.resolution is designed to be
                    # very similar to an entry in the features table)
                    fwhm = instrument.fwhm(instrumentname, w_obs, as_bounded=True)[
                        0
                    ] / (1.0 + redshift)

                    # decide if scalar (fixed) or tuple (fitted fwhm
                    # between upper and lower fwhm limits, happens in
                    # case of overlapping instruments)
                    if np.ma.is_masked(fwhm):
                        fwhm = fwhm[0]
                else:
                    fwhm = cleaned(row["fwhm"])

                self.fitter.add_feature_line(
                    name, cleaned(row["power"]), cleaned(row["wavelength"]), fwhm
                )

            elif kind == "dust_feature":
                self.fitter.add_feature_dust_feature(
                    name,
                    cleaned(row["power"]),
                    cleaned(row["wavelength"]),
                    cleaned(row["fwhm"]),
                )

            elif kind == "attenuation":
                self.fitter.add_feature_attenuation(name, cleaned(row["tau"]))

            elif kind == "absorption":
                self.fitter.add_feature_absorption(
                    name,
                    cleaned(row["tau"]),
                    cleaned(row["wavelength"]),
                    cleaned(row["fwhm"]),
                )

            else:
                raise PAHFITModelError(
                    f"Model components of kind {kind} are not implemented!"
                )

        self.fitter.finalize()

    @staticmethod
    def _parse_instrument_and_redshift(spec, redshift):
        """Get instrument redshift from Spectrum1D metadata or arguments."""
        # the rest of the implementation doesn't like Quantity...
        z = spec.redshift.value if redshift is None else redshift
        if z is None:
            # default of spec.redshift is None!
            z = 0

        inst = spec.meta["instrument"]
        if inst is None:
            raise PAHFITModelError("No instrument! Please set spec.meta['instrument'].")

        return inst, z
