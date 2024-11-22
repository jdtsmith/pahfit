from specutils import Spectrum1D
from astropy import units as u
from astropy import constants
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate, integrate

from pahfit import units
from pahfit.features.util import bounded_is_fixed, bounded_is_disabled
from pahfit.features import Features
from pahfit import instrument
from pahfit.errors import PAHFITModelError
from pahfit.fitters.ap_components import (BlackBody1D, ModifiedBlackBody1D,
                                          S07_attenuation, att_Drude1D)
from pahfit.fitters.ap_fitter import APFitter


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
        self.features.meta["redshift"] = z
        self.features.meta["instrument"] = inst

        # parse spectral data
        self.features.meta["user_unit"]["flux"] = spec.flux.unit
        _, _, _, lam, flux, _ = self._convert_spec_data(spec, z)
        lam_min = min(lam)
        lam_max = max(lam)

        # Some useful quantities for guessing
        median_flux = np.median(flux)
        Flambda = flux * units.intensity * (lam * units.wavelength) ** -2 * constants.c
        total_power = integrate.trapezoid(Flambda, lam * units.wavelength)

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
            w = lam_min + 0.1  # the wavelength used to compare
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
            lam_ref = np.clip(2898.0 / temp, lam_min, lam_max)
            bb = ModifiedBlackBody1D(1, temp)
            flux_ref = np.median(flux[(lam > lam_ref - 0.2) & (lam < lam_ref + 0.2)])
            amp_guess = flux_ref / bb(lam_ref)
            return np.clip(amp_guess / nbb, 0, 1.)

        loop_over_non_fixed("dust_continuum", "tau", dust_continuum_guess)

        def line_fwhm_guess(row):
            lam_line = row["wavelength"][0]
            if not instrument.within_segment(lam_line, inst):
                return 0

            fwhm = instrument.fwhm(inst, lam_line, as_bounded=True)[0][0]
            return fwhm

        def power_guess(row, fwhm):
            # local integration for the lines
            lam_line = row["wavelength"][0]
            if not instrument.within_segment(lam_line, inst):
                return 0

            factor = 1.5
            lam_min = lam_line - factor * fwhm
            lam_max = lam_line + factor * fwhm
            lam_window = (lam > lam_min) & (lam < lam_max)
            xpoints = lam[lam_window]
            ypoints = flux[lam_window]
            if np.count_nonzero(lam_window) >= 2:
                # difference between flux in window and flux around it
                Fnu_dlambda = integrate.trapezoid(flux[lam_window], lam[lam_window])
                # subtract continuum estimate, but make sure we don't go negative
                continuum = (ypoints[0] + ypoints[-1]) / 2 * (xpoints[-1] - xpoints[0])
                if continuum < Fnu_dlambda:
                    Fnu_dlambda -= continuum
            else:
                Fnu_dlambda = 0

            # this is an unphysical power (Fnu * dlambda), but we
            # convert to Fnu dnu = Fnu dnu/dlambda dlambda = Fnu c /
            # lambda **2 dlambda
            Fnu_dlambda *= units.intensity * units.wavelength
            Fnu_dnu = Fnu_dlambda * constants.c / (lam_line * units.wavelength) ** 2
            return Fnu_dnu.to(units.intensity_power).value

        def drude_power_guess(row):
            # multiply total power by some fraction to guess Drude power
            fwhm = row["fwhm"][0] * units.wavelength
            delta_w = spec.spectral_axis[-1] - spec.spectral_axis[0]
            return (total_power * fwhm / delta_w).to(units.intensity_power).value

        loop_over_non_fixed("dust_feature", "power", drude_power_guess)

        if integrate_line_flux:
            # calc line power using instrumental fwhm and integral over data
            loop_over_non_fixed("line", "power",
                                lambda row: power_guess(row, line_fwhm_guess(row)))
        else:
            loop_over_non_fixed("line", "power",
                                lambda row: median_flux * line_fwhm_guess(row))

        # Override the fwhms in the features table. Slightly different logic,
        # as the fwhm for lines are masked by default. TODO: leave FWHM
        # masked for lines, and instead have a sigma_v option. Any
        # requirements to guess and fit the line width, should be
        # encapsulated in sigma_v (the "broadening" of the line), as
        # opposed to fwhm which is the normal instrumental width.
        if calc_line_fwhm:
            for row_index in np.where(self.features["kind"] == "line")[0]:
                row = self.features[row_index]
                if row["fwhm"] is np.ma.masked:  # masked: overrideable by features table.
                    # A structured masked array is masked if _any_ of
                    # its elements is masked.  Table prevents setting
                    # values in such a masked array element, so we
                    # access the underlying array itself with .data
                    self.features["fwhm"].data[row_index]['val'] = line_fwhm_guess(row)
                    for b in ('min', 'max'):
                        self.features["fwhm"].data[row_index][b] = np.nan
                    self.features["fwhm"].data[row_index]['frozen'] = False
                elif not bounded_is_fixed(row["fwhm"]):
                    self.features["fwhm"].data[row_index]["val"] = line_fwhm_guess(row)

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

        self._set_up_fitter(inst, z, lam=x, use_instrument_fwhm=use_instrument_fwhm)
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

        for name in self.enabled_features:
            for column, value in self.fitter.get_result(name).items():
                try:
                    i = np.where(self.features["name"] == name)[0]
                    # do not update disabled attributes (e.g. line fwhm is usually masked)
                    if not bounded_is_disabled(self.features[column][i]):
                        self.features[column]["val"][i] = value
                    else:
                        self.features[column][i] = (value, np.nan, np.nan)
                except Exception as e:
                    print(f"Could not assign to attribute {name} in features table.")
                    print(f"Index {i=}")
                    print("Features table:", self.features)
                    raise e

    def plot(self, spec, redshift=None, use_instrument_fwhm=False,
             label_lines=False, scalefac_resid=2, update_fig=None,
             errorbar_kwargs=None, plot_kwargs=None, **kwargs):
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

        update_fig: matplotlib.pyplot.Figure
            Re-use limits from figure, to ease repetitive
            investigation of a sub-region.

        errorbar_kwargs : dict
            Customize the data points plot by passing a dictionary to
            be used as keyword arguments to
            :func:`matplotlib.pyplot.errorbar`.

        plot_kwargs : dict
            Additional keyword arguments to pass to
            :func:`matplotlib.pyplot.plot`.

        kwargs: dict
            Any additional keyword arguments are passed to
            :func:`matplotlib.pyplot.subplots` and must be valid
            keywords for this function.
        """
        instrument, z = self._parse_instrument_and_redshift(spec, redshift)
        _, _, _, lam, flux, unc = self._convert_spec_data(spec, z)
        enough_samples = max(10000, len(spec.wavelength))
        mnlam, mxlam = min(lam), max(lam)
        lam_mod = np.logspace(np.log10(mnlam), np.log10(mxlam), enough_samples)
        sp_unit = units.intensity.to_string().replace(" ", "")

        if update_fig:
            fig = update_fig
            if len(fig.axes) > 2:
                fig.axes[-1].remove()  # attenuation right axis
            axs = fig.axes
            limits = [{'xlim': ax.get_xlim(), 'ylim': ax.get_ylim()} for ax in axs]
            for ax in axs:
                ax.set_xscale("linear")  # avoid log warning
                ax.clear()
        else:
            fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 7),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True, **kwargs)

        # spectrum and best fit model
        ax = axs[0]
        ax.set_yscale("linear")
        ax.set_xscale("log")

        plot_kwargs = plot_kwargs or {}
        errorbar_kwargs = errorbar_kwargs or {}
        default_kwargs = dict(fmt="o", markeredgecolor="k", markerfacecolor="none",
                              ecolor="k", elinewidth=0.2, capsize=0.5, markersize=4)

        ax.errorbar(lam, flux, yerr=unc, **(default_kwargs | errorbar_kwargs))
        ax.set_ylim(0)
        ax.set_ylabel(r"$I_{\nu}$ " + f"({sp_unit})")
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", top="on", right="on",
                       direction="in", length=10)
        ax.tick_params(axis="both", which="minor", top="on", right="on",
                       direction="in", length=5)

        ext_model = None
        has_att = "attenuation" in self.features["kind"]
        if has_att:
            row = self.features[self.features["kind"] == "attenuation"][0]
            tau = row["tau"]['val']
            ext_model = S07_attenuation(tau_sil=tau)(lam_mod)

        has_abs = "absorption" in self.features["kind"]
        if has_abs:
            abs_model = np.ones_like(lam_mod)
            for fa in self.features[self.features['kind'] == "absorption"]:
                abs_func = att_Drude1D(tau=fa['tau']['val'],
                                       x_0=fa['wavelength']['val'],
                                       fwhm=fa['fwhm']['val'])
                abs_model *= abs_func(lam_mod)
            if ext_model is not None:
                ext_model *= abs_model
            else:
                ext_model = abs_model

        if ext_model is not None:
            ax_att = ax.twinx()  # y-axis for plotting the extinction curve
            ax_att.tick_params(which="minor", direction="in", length=5)
            ax_att.tick_params(which="major", direction="in", length=10)
            ax_att.minorticks_on()
            ln_att, = ax_att.plot(lam_mod, ext_model, "k--", alpha=0.7, **plot_kwargs,
                                  label='Attenuation & Absorption')
            ax_att.set_ylabel("Attenuation")
            ax_att.set_ylim(0, 1.1)
        else:
            ln_att = None
            ext_model = np.ones(len(lam_mod))

        # Define legend lines
        # Leg_lines = [
        #     mpl.lines.Line2D([0], [0], color="k", linestyle="--", lw=2),
        #     mpl.lines.Line2D([0], [0], color="#FE6100", lw=2),
        #     mpl.lines.Line2D([0], [0], color="#648FFF", lw=2, alpha=0.7),
        #     mpl.lines.Line2D([0], [0], color="#DC267F", lw=2, alpha=0.7),
        #     mpl.lines.Line2D([0], [0], color="#785EF0", lw=2, alpha=1),
        #     mpl.lines.Line2D([0], [0], color="#FFB000", lw=2, alpha=0.7),
        # ]

        # local utility
        def tabulate_components(kind):
            ss = {}
            for name in self.features[self.features["kind"] == kind]["name"]:
                ss[name] = self.tabulate(instrument, z, lam_mod,
                                         self.features["name"] == name)
            return {name: s.flux.value for name, s in ss.items()}

        total_cont = np.zeros_like(lam_mod)
        if "dust_continuum" in self.features["kind"]:
            # one plot for every component
            for i, y in enumerate(tabulate_components("dust_continuum").values()):
                ax.plot(lam_mod, y * ext_model, "#FFB000", alpha=0.7,
                        label=('Dust Continua' if i == 0 else None),
                        **plot_kwargs)

                # keep track of total continuum
                total_cont += y

        if "starlight" in self.features["kind"]:
            star_y = self.tabulate(instrument, z, lam_mod,
                                   self.features["kind"] == "starlight").flux.value
            ax.plot(lam_mod, star_y * ext_model, "magenta", alpha=0.7,
                    label='Stellar Continuum', **plot_kwargs)
            total_cont += star_y

        # now plot the dust bands and lines
        total_df = np.zeros_like(lam_mod)
        if "dust_feature" in self.features["kind"]:
            for i, y in enumerate(tabulate_components("dust_feature").values()):
                total_df += y
                ax.plot(lam_mod, (total_cont + y) * ext_model, "#648FFF",
                        label=('Dust Features' if i == 0 else None),
                        alpha=0.7, **plot_kwargs)

        if "line" in self.features["kind"]:
            for i, (name, y) in enumerate(tabulate_components("line").items()):
                ax.plot(lam_mod, (total_cont + y) * ext_model, "#DC267F",
                        label=('Lines' if i == 0 else None),
                        alpha=0.7, **plot_kwargs)
                if label_lines:
                    i = np.argmax(y)
                    # ignore out of range lines
                    if i > 0 and i < len(y) - 1:
                        w = lam_mod[i]
                        ax.text(w, y[i], name, va="center", ha="center",
                                rotation="vertical",
                                bbox=dict(facecolor="white", alpha=0.75, pad=0))

        # total continuum
        ax.plot(lam_mod, total_cont * ext_model, "#785EF0", alpha=1,
                label='Total Continuum', **plot_kwargs)

        ax.plot(lam_mod, self.tabulate(instrument, z, lam_mod).flux.value, "#11AA11",
                label='Model', alpha=1)

        handles = [ln for ln in ax.lines if not ln.get_label().startswith('_')]
        if ln_att:
            handles.insert(0, ln_att)
        ax.legend(handles=handles, prop={"size": 10}, loc="best")

        # residuals = data in rest frame - (model evaluated at rest frame wavelengths)
        res = flux - self.tabulate(instrument, 0, lam).flux.value
        std = np.nanstd(res)
        ax = axs[1]

        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.tick_params(axis="both", which="major", top="on", right="on",
                       direction="in", length=10)
        ax.tick_params(axis="both", which="minor", top="on", right="on",
                       direction="in", length=5)
        ax.minorticks_on()
        ax.axhline(0, linestyle="--", color="gray", zorder=0)
        ax.plot(lam, res, "ko", fillstyle="none", zorder=1,
                markersize=errorbar_kwargs.get("markersize", None),
                alpha=errorbar_kwargs.get("alpha", None),
                linestyle="none")
        ax.set_ylim(-scalefac_resid * std, scalefac_resid * std)
        ax.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax.set_ylabel(f"Residuals ({sp_unit})")

        # Refine x-axis
        ax.set_xlim(mnlam, mxlam)
        ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        fig.subplots_adjust(hspace=0)

        if update_fig:
            for lim, ax in zip(limits, axs):
                ax.set_xlim(*lim['xlim'])
                ax.set_ylim(*lim['ylim'])
        fig.tight_layout()

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
                lam_min, lam_max = ranges
            else:
                # In case of multiple ranges (multiple segments), choose
                # the min and max instead
                lam_min = min(r[0] for r in ranges)
                lam_max = max(r[1] for r in ranges)

            wfwhm = instrument.fwhm(instrumentname, lam_min, as_bounded=True)[0, 0]
            lam = np.arange(lam_min, lam_max, wfwhm / 2) * u.micron

        elif isinstance(wavelengths, Spectrum1D):
            lam = wavelengths.spectral_axis.to(u.micron) / (1 + z)
        else:
            # any other iterable will be accepted and converted to array
            lam = np.asarray(wavelengths) * u.micron

        # apply feature mask, make sub model, and set up functional
        if feature_mask is not None:
            features_to_use = self.features[feature_mask]
        else:
            features_to_use = self.features

        # if nothing is in range, return early with zeros
        if len(features_to_use) == 0:
            return Spectrum1D(
                spectral_axis=lam, flux=np.zeros(lam.shape) * u.dimensionless_unscaled
            )

        alt_model = Model(features_to_use)

        # Always use the current FWHM here (use_instrument_fwhm would
        # overwrite the value in the instrument overlap regions!)

        # need to wrap in try block to avoid bug: if all components are
        # removed (because of wavelength range considerations), it won't work
        try:
            alt_model._set_up_fitter(instrumentname, z, use_instrument_fwhm=False)
            fitter = alt_model.fitter
        except PAHFITModelError:
            return Spectrum1D(
                spectral_axis=lam, flux=np.zeros(lam.shape) * u.dimensionless_unscaled
            )

        flux_values = fitter.evaluate(lam.value)

        # apply unit stored in features table (comes from from last fit
        # or from loading previous result from disk)
        if "flux" not in self.features.meta["user_unit"]:
            flux_quantity = flux_values * u.dimensionless_unscaled
        else:
            user_unit = self.features.meta["user_unit"]["flux"]
            flux_quantity = (flux_values * units.intensity).to(user_unit)

        return Spectrum1D(spectral_axis=lam, flux=flux_quantity)

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
        self, instrumentname, redshift, lam=None, use_instrument_fwhm=True
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

        Parameters
        ----------

        instrumentname and redshift : needed to apply the instrument
        model and to determine which feature to exclude

        lam : array of observed wavelengths
            Used to exclude features from the model based on the actual
            observed data given.

        use_instrument_fwhm : bool
            When set to False, the instrument model is not used and the
            FWHM values are taken from the features table as-is. This is
            the current workaround to fit the widths of lines, until the
            "physical" and "instrumental" widths are conceptually
            separated (see issue #293).

        """
        # Fitting implementation can be changed by choosing another
        # Fitter class. TODO: make this configurable.
        self.fitter = APFitter()

        excluded = self._excluded_features(instrumentname, redshift, lam)
        self.enabled_features = self.features["name"][~excluded]

        def cleaned(value):
            val = value['val']
            if bounded_is_fixed(value):
                return val
            else:
                vmin = -np.inf if np.isnan(value['min']) else value['min']
                vmax = np.inf if np.isnan(value['max']) else value['max']
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
                # be careful with lines that have masked FWHM values here
                if use_instrument_fwhm or row["fwhm"] is np.ma.masked:
                    # One caveat here: redshift. We do the necessary
                    # adjustment as follows : 1. shift to observed wav;
                    # 2. evaluate fwhm at oberved wav; 3. shift back to
                    # rest frame wav (width in rest frame will be
                    # narrower than observed width)
                    lam_obs = row["wavelength"]["val"] * (1.0 + redshift)
                    # returned value is tuple (value, min, max). And
                    # min/max are already masked in case of fixed value
                    # (output of instrument.resolution is designed to be
                    # very similar to an entry in the features table)
                    calculated_fwhm = instrument.fwhm(
                        instrumentname, lam_obs, as_bounded=True
                    )[0] / (1.0 + redshift)

                    # decide if scalar (fixed) or tuple (fitted fwhm
                    # between upper and lower fwhm limits, happens in
                    # case of overlapping instruments)
                    if calculated_fwhm[1] is np.ma.masked:
                        fwhm = calculated_fwhm[0]
                    else:
                        fwhm = calculated_fwhm

                else:
                    # if instrument model is not to be used, just take
                    # the value as is specified in the Features table
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
