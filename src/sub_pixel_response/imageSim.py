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
from scipy.signal.windows import tukey
from scipy.special import legendre

from sub_pixel_response.simio import read_catalog, read_config

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
in_psf_oversam = 6
f_nu_ref = 3.631e-23 * (u.W / u.m**2) / u.Hz  # W/m^2/Hz
process_h = 4
process_v = 8
nside = 4088
std_pad = 24


def transform_pos(x, y, oversam=6):
    """Convert detector pixel coordinates into oversampled pixel space."""
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

    npad = int(np.ceil(tophatwidth + in_psf_oversam + 2))  # 6 from oversampling, 2 for safety margin
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
    """Compute PSF from polynomial PSF cube at given detector pixel location."""
    lpoly = l_poly_array(order, (pixloc[0] - 2043.5) / 2044.0, (pixloc[1] - 2043.5) / 2044.0)
    this_psf = smooth_and_pad(np.einsum("a,aij->ij", lpoly, inpsf_cube), tophatwidth=in_psf_oversam) / 64
    return this_psf


# More debugging prints
# print(os.getenv('SLURM_NTASKS'))
# print(os.getenv('SLURM_CPUS_PER_TASK'))
ncpu = int(os.getenv("SLURM_NTASKS", 1))


def sed_bb(w, T):
    """Return blackbody flux density at wavelength and temperature."""
    return (
        (8 * np.pi * const.h * const.c**2 / w**5) * 1 / (np.exp(const.h * const.c / (w * const.k_B * T)) - 1)
    ).decompose()


def convert_pos(ra, dec, wcs):
    """Convert RA/Dec to pixel coordinates using the World Coordinate System (WCS)."""
    worldCenter = galsim.CelestialCoord(ra=ra, dec=dec)
    imageCenter = wcs.posToImage(worldCenter)
    return (imageCenter.x, imageCenter.y)


def assign_star(x, y):
    """Assigning row of 8x4 processes to draw out stars"""
    # x_blue = np.clip(x // (nside // process_h), min = 0, max = 4088)
    # y_blue = np.clip(y // (nside // process_h), min = 0, max = 4088)
    x_blue_idx = int(np.clip(x // (nside // process_h), 0, process_h - 1))
    y_blue_idx = int(np.clip(y // (nside // process_v), 0, process_v - 1))
    task = y_blue_idx * process_h + x_blue_idx  # fixed this section, will see if this works
    return task


# j for given process number
def j_location(process, x_padding=0, y_padding=0):
    """Get tile bounding region in oversampled pixel coordinates."""
    xmin_j = (nside // process_h * (process % process_h)) * in_psf_oversam
    ymin_j = (nside // process_v * (process // process_h)) * in_psf_oversam
    xmax_j = (in_psf_oversam * nside // process_h) + xmin_j - 1
    ymax_j = (in_psf_oversam * nside // process_v) + ymin_j - 1
    process_bounds = galsim.BoundsI(
        xmin_j - x_padding, xmax_j + x_padding, ymin_j - y_padding, ymax_j + y_padding
    )
    return process_bounds


def draw_stars(
    j,
    cat,
    wcs,
    sca_num,
    nobj,
    is_in_circle,
    task_array,
    eff_area_table,
    transmission_curve,
    t_exp,
    roman_bandpasses,
    big_fft_params,
    x_padding=std_pad,
    y_padding=std_pad,
):  # Sorry, my code said all of these were undefined, I'll remove this after
    """Draw stars for tile index j into a temporary image section."""
    with fits.open("/users/PAS2340/karadiludovico/fits_files/psf_poly.fits") as inpsf_file:
        psf_data = np.copy(inpsf_file[sca_num].data[:, :, :])
    try:
        mybounds = j_location(j, x_padding=std_pad, y_padding=std_pad)
        tempImage = galsim.Image(bounds=mybounds, dtype=np.float32)
        for i in range(nobj):  # do I add nobj here? Or do I remove this completely?
            if task_array[i] != j:
                continue
            if not is_in_circle[i]:
                continue

            # First, calculating position
            degrees = galsim.AngleUnit(np.pi / 180)
            ra = cat["ra"][i] * degrees
            dec = cat["dec"][i] * degrees
            worldCenter = galsim.CelestialCoord(ra=ra, dec=dec)
            imageCenter = wcs.posToImage(worldCenter)
            new_image_center = transform_pos(imageCenter.x, imageCenter.y)
            imageCenter2 = galsim.PositionD(x=new_image_center[0], y=new_image_center[1])

            # Next, using position to compute PSF, use del command
            this_psf = compute_poly(psf_data, (new_image_center[0], new_image_center[1]))
            psf = galsim.Image(this_psf)
            # psf = galsim.roman.getPSF(sca_num, 'H158', SCA_pos=pos_SCA, wcs=mywcs,
            #     wavelength=roman_bandpasses['H158'])
            interp_psf = galsim.InterpolatedImage(
                psf, x_interpolant="lanczos32", scale=1
            )  # 0.11/in_psf_oversam)

            # Rest of flux calculations
            wav = np.arange(0.400, 2.600, 0.001) * u.um
            fluxUnnorm = sed_bb(wav, 5000 * u.K)
            fLambdaRef = f_nu_ref * const.c / wav**2
            mag = cat["mag_H"][i]
            norm = (
                10 ** (-0.4 * mag)
                * np.trapezoid(fLambdaRef * transmission_curve * wav, x=wav)
                / np.trapezoid(fluxUnnorm * transmission_curve * wav, x=wav)
            )
            flux = norm * fluxUnnorm
            nPhotQ = np.trapezoid(
                flux * eff_area_table["F158"] * u.m**2 * wav * t_exp / (const.h * const.c),
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

            st_model = galsim.DeltaFunction(flux=nPhot)
            source = galsim.Convolve([interp_psf, st_model], gsparams=big_fft_params)
            print("read flux calculations per star!", i, j)
            sys.stdout.flush()
            source.drawImage(tempImage, method="no_pixel", center=imageCenter2, add_to_image=True)
            print("read source.drawImage!")
            sys.stdout.flush()
            del psf, this_psf, interp_psf
        return tempImage
    except Exception as e:
        print(f"Error in process {j}: {str(e)}", file=sys.stderr)
        raise
    print(j_location)
    print("read draw_stars function and returned tempImage!")
    sys.stdout.flush()


# Main Execution
def run_simulation(config_path):
    """Main function to run the Roman star field image simulation.
    Parameters    ----------
    config_path : str
    Path to YAML configuration file with simulation parameters."""

    # Read configuration from YAML file
    config = read_config(config_path)
    print("read config!")
    sys.stdout.flush()

    # Read RA,Dec from star catalog
    # cat = galsim.Catalog(config['starCat'])
    cat = read_catalog(config["starCat"])
    # cat = cat[:1000] # added this line to only print out first 1000 stars in image
    nobj = len(cat["ra"])
    print("read cat!")
    sys.stdout.flush()

    degrees = galsim.AngleUnit(np.pi / 180)
    wcs_file_name = "/users/PCON0003/cond0007/PSF-TEST-FILES/Roman_WAS_simple_model_H158_13814_14.fits"
    read_image = galsim.fits.read(file_name=wcs_file_name, hdu=1, read_header=True)
    # mybounds = read_image.bounds
    read_image.header["CRVAL1"] = float(config["raCen"])
    read_image.header["CRVAL2"] = float(config["decCen"])
    read_image.header["LONPOLE"] = float(config["LONPOLE"])
    mywcs, neworigin = galsim.wcs.readFromFitsHeader(read_image.header)
    print(mywcs)
    print("read from degrees to mywcs, neworigin!")
    sys.stdout.flush()
    exit()

    # Determine which stars are in the circle
    if not config["randomPos"]:
        ra = cat["ra"] * np.pi / 180
        dec = cat["dec"] * np.pi / 180

        world_pos = mywcs.toWorld(galsim.PositionD(x=2044, y=2044))
        cos_theta = np.sin(world_pos.rad[1]) * np.sin(dec) + np.cos(world_pos.rad[1]) * np.cos(dec) * np.cos(
            ra - world_pos.rad[0]
        )
        is_in_circle = cos_theta > np.cos((0.11 * 4088) / (np.sqrt(2) * 3600) * np.pi / 180)
        print("read if not config randomPos for with fits.open starCat!")
        sys.stdout.flush()
    else:
        is_in_circle = np.ones(nobj, dtype=bool)
        print("read else statement for is_in_circle!")
        sys.stdout.flush()

    # Telescope exposure/SCA
    sca_num = int(config["SCA"])
    eff_area_table = aio.ascii.read(
        f"Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{sca_num:02d}_20240301.ecsv"
    )

    mirror_diameter = 2.37 * u.m
    geom_area = np.pi * mirror_diameter**2 / 4
    transmission_curve = eff_area_table["F158"] * u.m**2 / geom_area
    t_exp = 120 * u.s
    print("read from mirror_diameter to t_exp!")
    sys.stdout.flush()

    # Create output image with bounds
    xmin = ymin = -std_pad
    xmax = ymax = nside * in_psf_oversam + std_pad - 1
    out_image = galsim.Image(galsim.BoundsI(xmin, xmax, ymin, ymax))
    print("read out_image!")
    sys.stdout.flush()

    roman_bandpasses = galsim.roman.getBandpasses()
    big_fft_params = galsim.GSParams(maximum_fft_size=123000)
    print("read roman_bandpasses and big_fft_params!")
    sys.stdout.flush()

    # Tile/section assignment per star
    task_array = np.zeros(nobj, dtype=np.int32)

    for i in range(nobj):
        degrees = galsim.AngleUnit(np.pi / 180)
        ra = cat["ra"][i] * degrees
        dec = cat["dec"][i] * degrees
        x, y = convert_pos(ra, dec, mywcs)
        j = assign_star(x, y)
        task_array[i] = j

    # Prepare arguments for parallel processing
    multiprocess_stars = functools.partial(
        draw_stars,
        cat=cat,
        wcs=mywcs,
        sca_num=sca_num,
        nobj=nobj,
        is_in_circle=is_in_circle,
        task_array=task_array,
        eff_area_table=eff_area_table,
        transmission_curve=transmission_curve,
        t_exp=t_exp,
        roman_bandpasses=roman_bandpasses,
        big_fft_params=big_fft_params,
        x_padding=std_pad,
        y_padding=std_pad,
    )
    print("read multiprocess_stars!")
    sys.stdout.flush()

    # Determine number of processes to use
    num_processes = min(ncpu, nobj)

    # Printing number of processes to see if multiprocessing code above works
    print(num_processes)
    sys.stdout.flush()

    # Parallel processing and combine results
    with Pool(processes=num_processes) as pool:
        print("read with Pool statement!")
        sys.stdout.flush()
        for result in pool.imap_unordered(multiprocess_stars, range(num_processes)):
            print("read for result in pool statement!")
            sys.stdout.flush()
            if result is not None:
                print("read if results is not None!")
                sys.stdout.flush()
                out_image[
                    result.bounds
                ] += result  # Combine incrementally, having error with this line, changed it though
                print("read out_image += result!")
                sys.stdout.flush()
        print("ran pool for parallel processing!")
        sys.stdout.flush()

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
    out_image.write(config["outFile"])
    print("Image written to", config["outFile"])
    sys.stdout.flush()


# Main Execution
if __name__ == "__main__":
    run_simulation(sys.argv[1])
