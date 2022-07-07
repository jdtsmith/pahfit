"""Constants and enums used for the PAHFIT model calculation"""
from enum import IntEnum
import numpy as np

# Attenuation/absorption geometry type
class geometry(IntEnum):
    mixed = 0
    screen = 1

mbb_lam0 = 9.7 # scaling wavlength for modified blackbody (microns)

# Constants
hc_k = 1.4387769e4 # hc/k in micron K
bb_MJy_sr = 3.9728917e13 # Blackbody amplitude in MJy/sr
fwhmsig_2 = 2.77258872223978114 # (sigma_width of FWHM)^2/2 = 4*ln(2)
c = 2.99792458e14 # c in micron/s
gaussian_power_const = np.sqrt(np.pi/np.log(2))
