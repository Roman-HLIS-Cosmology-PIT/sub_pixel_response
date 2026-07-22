import datetime
import functools
import os
import sys
from multiprocessing import Pool

import astropy.io as aio
import galsim
import galsim.roman
import numpy as np
import pytz
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from furry_parakeet.pyimcom_croutines import gridG4460C
from scipy.signal.windows import tukey
from scipy.special import legendre

from sub_pixel_response.simio import read_catalog, read_config
from sub_pixel_response.utils.trapz import trapz

"""
Roman Telescope Star Field Image Simulator
-----------------------------------------

This script generates a simulated Roman Space Telescope image in the H158 band using a star catalog
from an imputed Besancon model and a polynomial PSF FITS cube. The Besancon model uses stellar coordinates
from the Galactic Buldge Time Domain Survey (GBTDS). For this image simulator, we are using SCA 14 from the
Roman telescope. The stars are drawn in parallel using an 8x4 toling pattern across the image.

Example YAML Configuration
-------------------------
raCen: 268.055873               # Pointing center RA (degrees)
decCen: -28.860960              # Pointing center Dec (degrees)
starCat: besancon_GB.fits       # Input FITS star catalog file with RA/Dec and H-Band magnitudes
SCA: 14                         # Detector used (1-18)
randomPos: false                # If true, it ignores the set RA/Dec values and randomly places stars on image
blackBody: true                 # Uses the black body flux model
outFile: "simulated.fits"       # Output file for simulated Roman star image
"""


def print_report(s):
    """Print message with UTC timestamp (debug helper)."""
    print(
        s, datetime.datetime.now(pytz.timezone("UTC")).strftime("%Y%m%d%H%M%S%f")
    )  # favorite format for time
    sys.stdout.flush()


# Global data and constants
GLB_DATA = {
    "in_psf_oversam": 6,
    "f_nu_ref": 3.631e-23 * (u.W / u.m**2) / u.Hz,  # W/m^2/Hz
    "process_h": 4,
    "process_v": 8,
    "nside": 4088,
    "furry_parakeet": True,
}
std_pad = 24  # C.H.: I'm waiting on moving this.


class GlobalContext:
    """
    Context manager for the global data.

    This is intended to be used in the form:

    .. code-block: python

        with GlobalContext({"nside": 2040}):

            # ... stuff with nside equal to 2040, e.g., if this were JWST data
            pass

        # nside will be set back to 4088 when you exit the "with"

    Parameters
    ----------
    modpars : dict
        Which parameters to change.

    """

    def __init__(self, modpars):
        self.modpars = modpars.copy()

    def __enter__(self):
        self.oldpars = GLB_DATA.copy()
        for k in self.modpars:
            if k not in GLB_DATA:
                raise ValueError("Tried to set a key that doesn't exist!")
            GLB_DATA[k] = self.modpars[k]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self.modpars:
            GLB_DATA[k] = self.oldpars[k]
        return False


def transform_pos(x, y, oversam=6):
    """
    Convert detector pixel coordinates into oversampled pixel space.

    Parameters
    ----------
    x : float
        Detector x-coordinate in native pixel units.
    y : float
        Detector y-coordinate in native pixel units.
    oversam : int, optional
        Oversampling factor used to map native detector pixels
        into the oversampled grid.

    Returns
    -------
    tuple of float
        Oversampled (X, Y) pixel coordinates.
    """
    X = oversam * (x - 0.5) + 0.5
    Y = oversam * (y - 0.5) + 0.5
    return (X, Y)


def smooth_and_pad(inArray: np.array, tophatwidth: float = 0.0) -> np.array:
    """
    Utility to smear a PSF with a tophat and apply a Tukey taper window.

    Parameters
    ----------
    inArray : np.array, shape : (ny, nx)
        Input PSF array to be smeared.
    tophatwidth : float, optional
        Width of the tophat convolution smearing PSF.

    Returns
    -------
    outArray : np.array, shape : (ny+npad*2, nx+npad*2)
        Smeared input PSF array.

    """

    npad = int(
        np.ceil(tophatwidth + GLB_DATA["in_psf_oversam"] + 2)
    )  # 6 from oversampling, 2 for safety margin
    npad += (4 - npad) % 4  # make a multiple of 4
    (ny, nx) = np.shape(inArray)
    nyy = ny + npad * 2
    nxx = nx + npad * 2
    outArray = np.zeros((nyy, nxx))
    outArray[npad:-npad, npad:-npad] = inArray
    outArrayFT = np.fft.fft2(outArray)

    # convolution (with Fourier Transform)
    uy = np.linspace(0, nyy - 1, nyy) / nyy
    uy = np.where(uy > 0.5, uy - 1, uy)
    ux = np.linspace(0, nxx - 1, nxx) / nxx
    ux = np.where(ux > 0.5, ux - 1, ux)
    outArrayFT *= np.sinc(ux[None, :] * tophatwidth) * np.sinc(uy[:, None] * tophatwidth)

    wy = np.fft.ifftshift(tukey(nyy, alpha=0.73))  # Tukey taper parameter derived from sampling relation:
    wx = np.fft.ifftshift(tukey(nxx, alpha=0.73))  # alpha = 1 - 2DP/(lambda*k)

    outArrayFT *= np.outer(wy, wx)

    outArray = np.real(np.fft.ifft2(outArrayFT))

    return outArray


def l_poly_array(PORDER, u_, v_):
    """
    Generates a length (PORDER+1)**2 array of the Legendre polynomials
    for n=0..PORDER { for m=0..PORDER { coef P_m(u_) P_n(v_) }}

    Parameters
    ----------
    PORDER : int
        >=0, order in each axis
    u_ : float
        x-position on chip scaled to -1..+1
    v_ : float
        y-position on chip scaled to -1..+1

    Returns
    -------
    arr: np.array, shape : ((PORDER+1)**2)
        the array of Legendre polynomial products
        constant (1) is first, then increasing x-order, then increasing y-order
    """

    ua = np.ones(PORDER + 1)
    va = np.ones(PORDER + 1)
    for m in range(1, PORDER + 1):
        L = legendre(m)
        ua[m] = L(u_)
        va[m] = L(v_)
    arr = np.outer(va, ua).flatten()
    return arr


def compute_poly(inpsf_cube, pixloc, order=1):
    """
    Compute PSF at a detector location from a polynomial PSF cube.

    Parameters
    ----------
    inpsf_cube : np.ndarray
        Polynomial PSF cube with dimensions
        ((order + 1)**2, ny, nx).
    pixloc : tuple of float
        Detector pixel location (x, y) where the PSF is evaluated.
    order : int, optional
        Maximum Legendre polynomial order used in the PSF model.

    Returns
    -------
    np.ndarray
        The interpolated and padded PSF evaluated at the
        specified detector location.
    """
    lpoly = l_poly_array(order, (pixloc[0] - 2043.5) / 2044.0, (pixloc[1] - 2043.5) / 2044.0)
    this_psf = (
        smooth_and_pad(np.einsum("a,aij->ij", lpoly, inpsf_cube), tophatwidth=GLB_DATA["in_psf_oversam"]) / 64
    )
    return this_psf


# More debugging prints
# print(os.getenv('SLURM_NTASKS'))
# print(os.getenv('SLURM_CPUS_PER_TASK'))
ncpu = int(os.getenv("SLURM_NTASKS", 1))


def sed_bb(w, T):
    """
    Return blackbody flux density at wavelength and temperature.

    Parameters
    ----------
    w : astropy.units.Quantity
        Wavelength array or value.
    T : astropy.units.Quantity
        Blackbody temperature.

    Returns
    -------
    astropy.units.Quantity
        Blackbody flux density evaluated at the specified
        wavelengths and temperature.
    """
    return (
        (8 * np.pi * const.h * const.c**2 / w**5) * 1 / (np.exp(const.h * const.c / (w * const.k_B * T)) - 1)
    ).decompose()


def convert_pos(ra, dec, wcs):
    """
    Convert RA/Dec to pixel coordinates using the World Coordinate System (WCS).

    Parameters
    ----------
    ra : float
        Right Ascension in radians.
    dec : float
        Declination in radians.
    wcs : galsim.WCS
        World Coordinate System object.

    Returns
    -------
    tuple of float
        Pixel coordinates (x, y) corresponding to the input RA/Dec.
    """
    worldCenter = galsim.CelestialCoord(ra=ra, dec=dec)
    imageCenter = wcs.posToImage(worldCenter)
    return (imageCenter.x, imageCenter.y)


def assign_star(x, y):
    """
    Assigning row of 8x4 processes to draw out stars.

    Parameters
    ----------
    x : float
        X-coordinate in pixel space.
    y : float
        Y-coordinate in pixel space.

    Returns
    -------
    int
        Process index (0-31) corresponding to the tile in which the star is located.
    """
    # x_blue = np.clip(x // (GLB_DATA["nside"] // GLB_DATA["process_h"]), min = 0, max = 4088)
    # y_blue = np.clip(y // (GLB_DATA["nside"] // GLB_DATA["process_h"]), min = 0, max = 4088)
    x_blue_idx = int(np.clip(x // (GLB_DATA["nside"] // GLB_DATA["process_h"]), 0, GLB_DATA["process_h"] - 1))
    y_blue_idx = int(np.clip(y // (GLB_DATA["nside"] // GLB_DATA["process_v"]), 0, GLB_DATA["process_v"] - 1))
    task = y_blue_idx * GLB_DATA["process_h"] + x_blue_idx
    return task


# j for given process number
def j_location(process, x_padding=0, y_padding=0):
    """
    Get tile bounding region in oversampled pixel coordinates.

    Parameters
    ----------
    process : int
        Process index (0-31) corresponding to the tile.
    x_padding : int, optional
        Padding in the x-direction for the bounding box.
    y_padding : int, optional
        Padding in the y-direction for the bounding box.

    Returns
    -------
    galsim.BoundsI
        Bounding box coordinates for the specified process.
    """
    xmin_j = (GLB_DATA["nside"] // GLB_DATA["process_h"] * (process % GLB_DATA["process_h"])) * GLB_DATA[
        "in_psf_oversam"
    ]
    ymin_j = (GLB_DATA["nside"] // GLB_DATA["process_v"] * (process // GLB_DATA["process_h"])) * GLB_DATA[
        "in_psf_oversam"
    ]
    xmax_j = (GLB_DATA["in_psf_oversam"] * GLB_DATA["nside"] // GLB_DATA["process_h"]) + xmin_j - 1
    ymax_j = (GLB_DATA["in_psf_oversam"] * GLB_DATA["nside"] // GLB_DATA["process_v"]) + ymin_j - 1
    process_bounds = galsim.BoundsI(
        xmin_j - x_padding, xmax_j + x_padding, ymin_j - y_padding, ymax_j + y_padding
    )
    return process_bounds


def draw_stars(
    j,
    cat,
    wcs,
    sca_num,
    task_array,
    eff_area_table,
    t_exp,
    roman_bandpasses,
    big_fft_params,
    psf_file,
    filter_name,
    x_padding=std_pad,
    y_padding=std_pad,
):
    """Draw stars for tile index j into a temporary image section."""
    with fits.open(psf_file) as inpsf_file:
        psf_data = np.copy(inpsf_file[sca_num].data[:, :, :])
    try:
        nobj = len(cat["ra"])
        mybounds = j_location(j, x_padding=std_pad, y_padding=std_pad)
        tempImage = galsim.Image(bounds=mybounds, dtype=np.float32)

        mirror_diameter = 2.37 * u.m
        geom_area = np.pi * mirror_diameter**2 / 4
        transmission_curve = eff_area_table[filter_name] * u.m**2 / geom_area

        # Moved these up here since they are constants, may help with speed improvements for code
        wav = np.arange(0.400, 2.600, 0.001) * u.um
        fluxUnnorm = sed_bb(wav, 5000 * u.K)
        fLambdaRef = GLB_DATA["f_nu_ref"] * const.c / wav**2

        for i in range(nobj):
            if task_array[i] != j:
                continue
            if "is_in_circle" in cat and not cat["is_in_circle"][i]:
                continue

            # First, calculating position
            degrees = galsim.AngleUnit(np.pi / 180)
            ra = cat["ra"][i] * degrees
            dec = cat["dec"][i] * degrees
            worldCenter = galsim.CelestialCoord(ra=ra, dec=dec)
            imageCenter = wcs.posToImage(worldCenter)
            new_image_center = transform_pos(imageCenter.x, imageCenter.y)
            imageCenter2 = galsim.PositionD(x=new_image_center[0], y=new_image_center[1])

            # Rest of flux calculations
            mag = cat["mag_H"][i]
            norm = (
                10 ** (-0.4 * mag)
                * trapz(fLambdaRef * transmission_curve * wav, x=wav)
                / trapz(fluxUnnorm * transmission_curve * wav, x=wav)
            )
            flux = norm * fluxUnnorm
            nPhotQ = trapz(
                flux * eff_area_table[filter_name] * u.m**2 * wav * t_exp / (const.h * const.c),
                x=wav,
            )
            nPhotQ = nPhotQ.decompose()
            nPhot = nPhotQ.value
            if not np.isfinite(nPhot):
                print(
                    f"!! WARNING (j={j}, i={i}): Invalid flux calculated: {nPhot}",
                    flush=True,
                )
                continue  # Skip this star
            if not mybounds.includes(imageCenter2):
                print(
                    f"!! WARNING (j={j}, i={i}): Star position {imageCenter2} is outside bounds {mybounds}",
                    flush=True,
                )
                continue  # Skip this star

            # Next, using position to compute PSF, use del command
            this_psf = compute_poly(psf_data, (new_image_center[0], new_image_center[1]))
            star = galsim.Image(this_psf * nPhotQ)
            # psf = galsim.roman.getPSF(sca_num, 'H158', SCA_pos=pos_SCA, wcs=mywcs,
            #     wavelength=roman_bandpasses['H158'])
            if GLB_DATA["furry_parakeet"]:
                interp_star_array = np.zeros((1, np.size(star.array)), dtype=np.float64)
                x_nearest_int = round(new_image_center[0])
                y_nearest_int = round(new_image_center[1])
                delta_x = new_image_center[0] - x_nearest_int
                delta_y = new_image_center[1] - y_nearest_int
                (ny, nx) = np.shape(star.array)
                xarray = np.linspace(0, nx - 1, nx)
                yarray = np.linspace(0, ny - 1, ny)
                gridG4460C(star.array, xarray[None, :] - delta_x, yarray[None, :] - delta_y, interp_star_array)
                # tempImage.array[center][center] += interp_star_array
                xc = x_nearest_int - nx // 2
                yc = y_nearest_int - ny // 2
                dy1 = max(0, -yc)
                dy2 = min(ny, tempImage.array.shape[0] - yc)
                dx1 = max(0, -xc)
                dx2 = min(nx, tempImage.array.shape[1] - xc)
                tempImage.array[yc + dy1 : yc + dy2, xc + dx1 : xc + dx2] += interp_star_array[dy1:dy2, dx1:dx2]
                del star, this_psf, interp_star_array
            else:
                interp_star = galsim.InterpolatedImage(
                    star, x_interpolant="lanczos32", scale=1
                )  # 0.11/in_psf_oversam)
                interp_star.drawImage(tempImage, method="no_pixel", center=imageCenter2, add_to_image=True)
                del star, this_psf, interp_star
        return tempImage
    except Exception as e:
        print(f"Error in process {j}: {str(e)}", file=sys.stderr)
        raise


# Main Execution
def run_simulation(config_path):
    """
    Main function to run the Roman star field image simulation.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file with simulation parameters.
    """

    # Read configuration from YAML file
    config = read_config(config_path)

    # Read RA,Dec from star catalog
    # cat = galsim.Catalog(config['starCat'])
    cat = read_catalog(config["starCat"])
    # cat = cat[:1000] # added this line to only print out first 1000 stars in image
    nobj = len(cat["ra"])

    degrees = galsim.AngleUnit(np.pi / 180)
    WCSSTRING = "\n".join(
        [
            "XTENSION= 'IMAGE   '           / Image extension                                ",
            "BITPIX  =                  -64 / array data type                                ",
            "NAXIS   =                    2 / number of array dimensions                     ",
            "NAXIS1  =                 4088                                                  ",
            "NAXIS2  =                 4088                                                  ",
            "PCOUNT  =                    0 / number of parameters                           ",
            "GCOUNT  =                    1 / number of groups                               ",
            "EXPTIME =                139.8                                                  ",
            "MJD-OBS =         62471.492045                                                  ",
            "DATE-OBS= '2029-12-01 11:48:32.688000'                                          ",
            "FILTER  = 'H158    '                                                            ",
            "ZPTMAG  =   16.800870916182618                                                  ",
            "GS_XMIN =                    1 / GalSim image minimum x coordinate              ",
            "GS_YMIN =                    1 / GalSim image minimum y coordinate              ",
            "GS_WCS  = 'GSFitsWCS'          / GalSim WCS name                                ",
            "CTYPE1  = 'RA---TAN-SIP'                                                        ",
            "CTYPE2  = 'DEC--TAN-SIP'                                                        ",
            "CRPIX1  =               2044.0                                                  ",
            "CRPIX2  =               2044.0                                                  ",
            "CD1_1   = 3.01922901086850E-05                                                  ",
            "CD1_2   = 1.45559107465136E-06                                                  ",
            "CD2_1   = -8.3872456214526E-07                                                  ",
            "CD2_2   = 2.93576778237758E-05                                                  ",
            "CUNIT1  = 'deg     '                                                            ",
            "CUNIT2  = 'deg     '                                                            ",
            "CRVAL1  =   10.208584415642562                                                  ",
            "CRVAL2  =   -44.33853770184239                                                  ",
            "A_ORDER =                    4                                                  ",
            "A_0_2   =      3.851828071E-10                                                  ",
            "A_0_3   =      5.492409696E-14                                                  ",
            "A_0_4   =      3.825353128E-18                                                  ",
            "A_1_1   =     -1.232185377E-09                                                  ",
            "A_1_2   =     -9.743979693E-14                                                  ",
            "A_1_3   =       2.66249338E-17                                                  ",
            "A_2_0   =      3.802404353E-10                                                  ",
            "A_2_1   =     -9.031463862E-14                                                  ",
            "A_2_2   =     -6.271302544E-17                                                  ",
            "A_3_0   =      2.325088216E-14                                                  ",
            "A_3_1   =      2.521067326E-17                                                  ",
            "A_4_0   =      1.425534054E-17                                                  ",
            "B_ORDER =                    4                                                  ",
            "B_0_2   =     -1.175573884E-09                                                  ",
            "B_0_3   =      1.303875779E-14                                                  ",
            "B_0_4   =      1.602230927E-17                                                  ",
            "B_1_1   =     -1.793186122E-11                                                  ",
            "B_1_2   =     -1.532973486E-13                                                  ",
            "B_1_3   =     -3.870326104E-17                                                  ",
            "B_2_0   =      5.982538571E-11                                                  ",
            "B_2_1   =      1.443685076E-13                                                  ",
            "B_2_2   =      1.727380843E-17                                                  ",
            "B_3_0   =      2.897221014E-14                                                  ",
            "B_3_1   =      2.713725388E-17                                                  ",
            "B_4_0   =      1.591294122E-18                                                  ",
            "EQUINOX =               2000.0                                                  ",
            "WCSAXES =                    2                                                  ",
            "WCSNAME = 'wfiwcs_20210204_d2'                                                  ",
            "TELESCOP= 'Roman   '                                                            ",
            "INSTRUME= 'WFC     '                                                            ",
            "RA_TARG =               10.489                                                  ",
            "DEC_TARG=             -44.4299                                                  ",
            "PA_OBSY =  -118.00999999999999                                                  ",
            "PA_FPA  =   1.9899999999999998                                                  ",
            "SCA_NUM =                   14                                                  ",
            "ORIENTAT=   2.1863008787824225                                                  ",
            "LONPOLE =                180.0                                                  ",
            "SKY_MEAN=                 74.0                                                  ",
            "EXTNAME = 'SCI     '           / extension name                                 ",
            "EXTVER  =                    1 / extension value                                ",
        ]
    )
    myheader = fits.Header.fromstring(
        WCSSTRING,
        sep="\n",
    )
    # mybounds = read_image.bounds
    myheader["CRVAL1"] = float(config["raCen"])
    myheader["CRVAL2"] = float(config["decCen"])
    myheader["LONPOLE"] = float(config["LONPOLE"])
    mywcs = galsim.AstropyWCS(header=myheader)
    # exit()
    # K.D. : I commented out exit() for right now because it stops the job from executing and running

    # Determine which stars are in the circle
    if not config["randomPos"]:
        ra = cat["ra"] * np.pi / 180
        dec = cat["dec"] * np.pi / 180

        world_pos = mywcs.toWorld(galsim.PositionD(x=2044, y=2044))
        cos_theta = np.sin(world_pos.rad[1]) * np.sin(dec) + np.cos(world_pos.rad[1]) * np.cos(dec) * np.cos(
            ra - world_pos.rad[0]
        )
        is_in_circle = cos_theta > np.cos((0.11 * 4088) / (np.sqrt(2) * 3600) * np.pi / 180)
        cat["is_in_circle"] = is_in_circle

    # Telescope exposure/SCA
    sca_num = int(config["SCA"])
    eff_area_table = aio.ascii.read(
        f"Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{sca_num:02d}_20240301.ecsv"
    )

    t_exp = 120 * u.s

    # Create output image with bounds
    xmin = ymin = -std_pad
    xmax = ymax = GLB_DATA["nside"] * GLB_DATA["in_psf_oversam"] + std_pad - 1
    out_image = galsim.Image(galsim.BoundsI(xmin, xmax, ymin, ymax))

    roman_bandpasses = galsim.roman.getBandpasses()
    big_fft_params = galsim.GSParams(maximum_fft_size=123000)

    # Tile/section assignment per star
    task_array = np.zeros(nobj, dtype=np.int32)

    for i in range(nobj):
        degrees = galsim.AngleUnit(np.pi / 180)
        ra = cat["ra"][i] * degrees
        dec = cat["dec"][i] * degrees
        x, y = convert_pos(ra, dec, mywcs)
        j = assign_star(x, y)
        task_array[i] = j

    # Filter read from configuration file
    filter_name = config["FILTER"]

    # Read PSF file from configuration file
    psf_file = config["PSFFILE"]

    # Prepare arguments for parallel processing
    multiprocess_stars = functools.partial(
        draw_stars,
        cat=cat,
        wcs=mywcs,
        sca_num=sca_num,
        task_array=task_array,
        eff_area_table=eff_area_table,
        t_exp=t_exp,
        roman_bandpasses=roman_bandpasses,
        big_fft_params=big_fft_params,
        x_padding=std_pad,
        y_padding=std_pad,
        psf_file=psf_file,
        filter_name=filter_name,
    )

    # Determine number of processes to use
    num_processes = min(ncpu, nobj)

    # Printing number of processes to see if multiprocessing code above works
    # print(num_processes)
    # sys.stdout.flush()

    # Parallel processing and combine results
    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(multiprocess_stars, range(num_processes)):
            if result is not None:
                out_image[result.bounds] += result

    # Add results to blank canvas using numpy zeros
    """final_image = np.zeros((24528, 24528))
    for j in range(len(result)):
        bounds = j_location(j)
        xmin = bounds.getXMin()
        xmax = bounds.getXMax()
        ymin = bounds.getYMin()
        ymax = bounds.getYMax()
        final_image[xmin:xmax, ymin:ymax] = result[j]
        print('read for j statement with final_image and results for j!')
        sys.stdout.flush()"""

    # out_image = process_func(0)

    # May remove this print statement and sys.stdout.flush() later, but for now it is useful to see if it's
    # written to the right image/file
    out_image.write(config["outFile"])
    print("Image written to", config["outFile"])
    sys.stdout.flush()


# Main Execution
if __name__ == "__main__":
    run_simulation(sys.argv[1])
