"""Constants and enums used for the PAHFIT model calculation"""
from enum import IntEnum
import numpy as np

# Validity range multipliers (each side)


# integral_(-2.25 2 (σ sqrt(2 log(2))))^(2.25 2 (σ sqrt(2 log(2)))) e^(-x^2/(2 σ^2))/(sqrt(2 π) σ) dx =
#   erf(4.5 sqrt(log(2))) ≈ 1.16857... × 10^-7
validity_gaussian_fwhms = 2.25  # = 5.3 sigma


# Note: Drude's have enough power outside of 1-100um that they do not
# have a validity range
#  validity_drude_fwhms = 50


# Parameter type
class param_type(IntEnum):
    independent = -2
    fixed = -1

# Attenuation/absorption geometry type
class geometry(IntEnum):
    mixed = 0
    screen = 1


mbb_lam0 = 9.7  # scaling wavelength for modified blackbody (microns)

# Constants
hc_k = 1.4387769e4  # hc/k in micron K
bb_MJy_sr = 3.9728917e13  # Blackbody amplitude in MJy/sr
fwhmsig_2 = 2.77258872223978114  # (sigma_width of FWHM)^2/2 = 4*ln(2)
c = 2.99792458e14  # c in micron/s
gaussian_power_const = np.sqrt(np.pi / np.log(2))
