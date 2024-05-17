"""
pahfit.model.components: Functional Forms for the PAHFIT Model
"""
import numpy as np
from .const import bb_MJy_sr, hc_k, mbb_lam0, fwhmsig_2, gaussian_power_const, c
from .pfnumba import pahfit_jit


@pahfit_jit
def blackbody(lam, tau, T):
    """Calculate the blackbody function B_nu(lambda, T).

    Arguments:
    ----------
    lam: wavelength vector in microns.

    tau: The opacity/amplitude of the blackbody.  Unitless if spectra
      are in MJy/sr.

    T: BB temperature in K.
    """
    return bb_MJy_sr * tau / lam**3 / (np.exp(hc_k / lam / T) - 1.)


@pahfit_jit
def modified_blackbody(lam, tau, T):
    """Calculate modified blackbody.  See blackbody."""
    return blackbody(lam, tau, T) * (mbb_lam0 / lam)**2

# --------------------------------------------------------------------
#   A NOTE ON INTEGRATED INTENSITY/POWER OF GAUSSIAN/DRUDE PROFILES
#
# Because of the IR convention of mixed-optical+radio-units,
# i.e. f_nu(lambda), there are some practical considerations related
# to integrals over Gaussian and Drude profiles.
#
# These two model components have finite widths which are expressed in
# real wavelength units (microns).  As a result, both are naturally
# expressed as a function of wavelength.  Their normalizing amplitudes
# (A) must then, necessarily, be given *per unit wavelength* (A_lam),
# so that integrals of the features over wavelength yield no residual
# wavelength unit dependence (e.g. from W/m^2/micron -> W/m^2).

# The full integrated "power", P, of the Drude and Gaussian over
# *wavelength* are quite similar.  Given their FWHM and an amplitude
# per unit wavelength, A_lam:
#
#   P = A_lam * fwhm/2 * (pi, sqrt(pi/ln(2))) for (Drude, Gaussian)
#
# I.e. the Drude of the same FWHM contains more power than the
# Gaussian, by a factor of 1.476.
#
# But there is one more important detail.  During the fit, amplitudes
# A are varied so as to match an implicit *f_nu* comparison spectrum
# (due to the mixed IR unit convention), at position lam_0.  The
# fitted amplitude is therefore of the A_nu variety, not the (desired)
# A_lam.
#
# Since:
#
#  A_nu dnu = A_lam dlam => A_lam = -A_nu c/lam^2
#
# the full power, using the actual fitted amplitude A_nu (and
# reversing integration limits to remove the minus sign), is then:
#
#   P = A_nu c/lam_0^2 * fwhm/2 * (pi, sqrt(pi/ln(2)))
#
# Note: a feature of precisely the same width and (f_nu) amplitude has
# a smaller amount of power when it appears at a longer wavelength!
# Our Drude formulation reflects these mixed units, just as would
# expressing the Blackbody function per unit frequency in terms of
# wavelength B_nu(lambda).
#
# --------------------------------------------------------------------
#   SCALED POWER
#
# Internally to pahfit.model, instead of real power, we use
# "scaled power", SP:
#
#   SP = P * lam_0/c
#
# Scaled power has identical units to the (f_nu-flavored) input
# spectrum, and similar values.  On output of fitting results, true
# power is restored.
# --------------------------------------------------------------------


@pahfit_jit
def drude(lam, P_or_A, lam_0, fwhm, amp=False, scaled=True):
    """Calculate the Drude function D_lam(A_nu, lam, lam_0, FWHM).

    Parameters
    ----------

    lam : float
        Wavelength vector over which to compute the Drude (microns)

    p_or_a : float
        Either power (by default, "scaled") or amplitude (see
        `power').

    lam_0 : float
        Central wavelength of the feature (microns).

    fwhm : float
        full-width at half maximum of the feature (microns).

    amp : bool, optional
        True if an amplitude is passed instead of scaled power
        (default: False).

    scaled : bool, optional
        If power is passed, whether it is "scaled power" (see `power';
        default: True).

    Returns
    -------

    A vector array containing the full Drude profile over the input
    wavelengths.
    """
    frac_fwhm = fwhm / lam_0
    if not amp:
        P_or_A = amplitude(P_or_A, lam_0, fwhm, drude=True, scaled=scaled)
    return P_or_A * frac_fwhm**2 / ((lam / lam_0 - lam_0 / lam)**2 + frac_fwhm**2)


@pahfit_jit
def gaussian(lam, P_or_A, lam_0, fwhm, amp=False, scaled=True):
    """Calculate the Gaussian Function G_lam(A_nu, lam, lam_0, FWHM).
    See `drude` for details on the arguments.
    """
    if not amp:
        P_or_A = amplitude(P_or_A, lam_0, fwhm, scaled=scaled)
    return P_or_A * np.exp(-(lam - lam_0)**2 * fwhmsig_2 / fwhm**2)


@pahfit_jit
def power(amplitude, lam_0, fwhm, drude=False, scaled=False):
    """Return the power (aka integrated intensity) given the
    amplitude, central wavelength, and FWHM of a Gaussian or Drude
    profile.

    Parameters
    ----------

    amplitude: The amplitude of the feature, in the units of the
        input spectrum.

    lam_0 : float
        The central wavelength (microns).

    fwhm : float
        The full-width at half maximum of the feature (microns).

    drude : (bool, optional)  Whether this is a Drude feature (default:
        False).

    scaled : bool, optional
        Whether to return "Scaled Power" (see below, default: False).

    Returns
    -------

    power : float
        Integrated power of the feature (units: ``[amplitude] * Hz``),
        or scaled power (units: ``[amplitude]``).

    .. note::

           *Scaled Power* is power scaled into the flux density units of
           the input spectrum (e.g. mJy, MJy/sr, etc.): ``SP = P lam_0/c``
           An interpretation of scaled power: if a feature has a FWHM
           similar to its central wavelength, the scaled power is
           approximately the feature's peak amplitude.  Scaled power
           is used internally for fitting, so as to avoid large
           mismatch in numeric scale between powers and feature
           amplitudes.

    See Also
    --------

    amplitude : Convert (scaled) power to amplitude.
    """
    P = amplitude / lam_0 * fwhm / 2 * (np.pi if drude else gaussian_power_const)
    if not scaled:
        P *= c / lam_0
    return P


@pahfit_jit
def amplitude(power, lam_0, fwhm, drude=False, scaled=False):
    """Return the amplitude corresponding to a given **scaled power**.

    Returns
    -------

    amplitude : float
        Amplitude of the feature in f_nu units.

    See Also
    --------

    power : power from amplitude; see for parameter definition.
    """
    A = power * lam_0 / fwhm * 2 / (np.pi if drude else gaussian_power_const)
    if not scaled:
        A *= lam_0 / c
    return A
