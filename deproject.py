## https://github.com/astrojysun/Sun_Astro_Tools/blob/master/sun_astro_tools/coord.py

from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from astropy.table import Table, Column, Row
import pandas as pd
from math import sqrt


def deproject(center_coord=None, incl=0*u.deg, pa=0*u.deg,
              header=None, wcs=None, naxis=None, ra=None, dec=None,
              return_offset=False):

    """
    Calculate deprojected radii and projected angles in a disk.
    This function deals with projected images of astronomical objects
    with an intrinsic disk geometry. Given sky coordinates of the
    disk center, disk inclination and position angle, this function
    calculates deprojected radii and projected angles based on
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coodinates (`ra` + `dec`).
    Both deprojected radii and projected angles are defined relative
    to the center in the inclined disk frame. For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.
    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` object or 2-tuple
        Sky coordinates of the disk center
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    header : `~astropy.io.fits.Header` object, optional
        FITS header specifying the WCS and size of the output 2D maps
    wcs : `~astropy.wcs.WCS` object, optional
        WCS of the output 2D maps
    naxis : array-like (with two elements), optional
        Size of the output 2D maps
    ra : array-like, optional
        RA coordinate of the sky locations of interest
    dec : array-like, optional
        DEC coordinate of the sky locations of interest
    return_offset : bool, optional
        Whether to return the angular offset coordinates together with
        deprojected radii and angles. Default is to not return.
    Returns
    -------
    deprojected coordinates : list of arrays
        If `return_offset` is set to True, the returned arrays include
        deprojected radii, projected angles, as well as angular offset
        coordinates along East-West and North-South direction;
        otherwise only the former two arrays will be returned.
    Notes
    -----
    This is the Python version of an IDL function `deproject` included
    in the `cpropstoo` package. See URL below:
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
    """

    if isinstance(center_coord, SkyCoord):
        x0_deg = center_coord.ra.degree
        y0_deg = center_coord.dec.degree
    else:
        x0_deg, y0_deg = center_coord
        if hasattr(x0_deg, 'unit'):
            x0_deg = x0_deg.to(u.deg).value
            y0_deg = y0_deg.to(u.deg).value
    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    if header is not None:
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    elif (wcs is not None) and (naxis is not None):
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    else:
        ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        if hasattr(ra_deg, 'unit'):
            ra_deg = ra_deg.to(u.deg).value
            dec_deg = dec_deg.to(u.deg).value

    # recast the ra and dec arrays in term of the center coordinates
    # arrays are now in degrees from the center
    dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dy_deg = dec_deg - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi/2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) +
                    dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) -
                    dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_offset:
        return radius_deg, projang_deg, dx_deg, dy_deg
    else:
        return radius_deg, projang_deg
    
# to get galaxy data from sample table
def get_galaxy_specs(galaxy = 'NGC0628'):
    
    from astropy.table import Table, Column, Row
    import pandas as pd
    #     rows = [5, 16, 108, 109]
    fits_filename = 'phangs_sample_table_v1p1.fits'
    sample_tab = Table.read(fits_filename, format='fits')
    df = sample_tab.to_pandas()
    names = np.array([str(x)[2:-1].strip() for x in df['NAME']], dtype=str)
    df['NAMES']=names
    RA = (df['RA_DEG'][df['NAMES'] == galaxy]).to_numpy()[0]
    DEC = (df['DEC_DEG'][df['NAMES'] == galaxy]).to_numpy()[0]
    POSANG = (df['POSANG'][df['NAMES'] == galaxy]).to_numpy()[0]
    INCL = (df['INCL'][df['NAMES'] == galaxy]).to_numpy()[0]
    DIST = (df['DIST'][df['NAMES'] == galaxy]).to_numpy()[0]
    
    return RA, DEC, POSANG, INCL, DIST

def fit_to_normal(x, y, xscale='log', yscale='lin', cut=2e1):
    
    from scipy.optimize import curve_fit
    # scale data
    if xscale=='log':
        if cut=='None':
            cut=0
        else:
            cut = np.log10(cut)
        x = np.log10(x)
    else:
        if cut=='None':
            cut=0
    if yscale=='log':
        y = np.log10(y)
        
    # make normal distribution
    def gauss(x,mu,sigma,A):
        return A*np.exp(-((x-mu)/sigma)**2/2)

    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1) + gauss(x,mu2,sigma2,A2)
    if cut==0:
        # fit to normal
        p0 = (np.log10(5e-1), np.log10(1e-1), 0.003)
        p1, pcov1 = curve_fit(gauss, x, y, p0)
        fit_y1 = gauss(x, *p1) 
        p0 = (np.log10(5e-1), np.log10(1e-1), 0.003)
        p2, pcov2 = curve_fit(gauss, x, y, p0)
        fit_y2 = gauss(x, *p2) 
        residuals1 = y - fit_y1
        residuals2 = y - fit_y2
    else:
        # fit to normal
        p0 = (np.log10(2), np.log10(3), 0.2)
        p1, pcov1 = curve_fit(gauss, x[x<cut], y[x<cut], p0)
        fit_y1 = gauss(x, *p1) 
        p0 = (np.log10(50), np.log10(100), 1)
        p2, pcov2 = curve_fit(gauss, x[x>=cut], y[x>=cut], p0)
        fit_y2 = gauss(x, *p2) 
        residuals1 = y - fit_y1
        residuals2 = y - fit_y2
    # FWHM = 2 sqrt(2 ln (2)) * sigma[sd] = 2.35*sigma

    fits = {
        "mu1": p1[0], # peak or mean
        "sd1": p1[1], # standard deviation
        "mu_err1": np.sqrt(np.diag(pcov1))[0], # 1 stddex err
        "sd_err1": np.sqrt(np.diag(pcov1))[1], # 1 stddex err
        "mu2": p2[0], # peak or mean
        "sd2": p2[1], # standard deviation
        "perr2": np.sqrt(np.diag(pcov2)), # 1 stddex err
        "fwhm": 2.35*p1[1], # full width at half max
        "x": x, # x data (transformed)
        "y": y, # y data (transformed)
        "fit_y1": fit_y1, # fitted y data
        "fit_y2": fit_y2, # fitted y data
        "residuals1": residuals1, # y residuals
        "residuals2": residuals2, # y residuals
        }
    return fits


def rms_val(arr):
    return sqrt(sum(n*n for n in arr)/len(arr))


# smooth
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth




def fit_to_linear(x, y, cut=(0, 1e4)):
    
    from scipy.optimize import curve_fit
      
    # make straight line: y = mx + c
    def linear(x, m, c):
        return m*x + c
    
    y = y[(x>=cut[0]) & (x<=cut[1])]
    x = x[(x>=cut[0]) & (x<=cut[1])]

    p1, pcov = curve_fit(linear, x, y)
    X = np.linspace(cut[0]-0.5, cut[1]+0.5, 100)
    fit_y = linear(X, *p1) 
    perr = np.sqrt(np.diag(pcov))

#     residuals = y - fit_y
    fit_slope = p1[0]
    fit_intercept_at_1MJy_str = p1[1]

    fit = {
        "x": X, # x data (transformed)

        "fit_y": fit_y, # fitted y data

        "slope": fit_slope, # slope m - from fit
        
        "intercept": fit_intercept_at_1MJy_str, # intercept c - from fit
        
        "perr": perr
        }
    return fit



