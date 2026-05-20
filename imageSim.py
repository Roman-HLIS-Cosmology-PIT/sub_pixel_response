import numpy as np
import galsim
import galsim.roman
from astropy.table import Table
import yaml
import pandas as pd
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
import astropy.io as aio
from astropy import constants as const
import warnings
import argparse
from astropy.io import fits
import sys
from multiprocessing import Pool, cpu_count
import functools
import os
from scipy.special import legendre
from scipy.signal.windows import tukey
import datetime
import pytz

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
fNuRef = 3.631e-23 * (u.W / u.m**2) / u.Hz  # W/m^2/Hz
process_h = 4
process_v = 8
nside = 4088
std_pad = 24
print("read global data!")
sys.stdout.flush()


def transformPos(x, y, oversam=6):
    """Convert detector pixel coordinates into oversampled pixel space."""
    X = oversam * (x - 0.5) + 0.5
    Y = oversam * (y - 0.5) + 0.5
    return (X, Y)


print("transformPos read!")
sys.stdout.flush()


def smooth_and_pad(
    inArray: np.array, tophatwidth: float = 0.0
) -> np.array:
    """
    Utility to smear a PSF with a tophat and apply a Tukey taper window.

    Parameters
    ----------
    inArray : np.array, shape : (ny, nx)
        Input PSF array to be smeared.
    tophatwidth : float, optional

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
    outArrayFT *= (
        np.sinc(ux[None, :] * tophatwidth)
        * np.sinc(uy[:, None] * tophatwidth)
    )
    
    wy = np.fft.ifftshift(tukey(nyy, alpha=0.73))   # Tukey taper parameter derived from sampling relation:
    wx = np.fft.ifftshift(tukey(nxx, alpha=0.73))   # alpha = 1 - 2DP/(lambda*k)

    outArrayFT *= np.outer(wy, wx)

    outArray = np.real(np.fft.ifft2(outArrayFT))

    return outArray


print("returned outArray and read smooth_and_pad function!")
sys.stdout.flush()


def LPolyArr(PORDER, u_, v_):
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


print("returned LPolyArr!")
sys.stdout.flush()


def compute_poly(inpsf_cube, pixloc, order=1):
    """Compute PSF from polynomial PSF cube at given detector pixel location."""
    lpoly = LPolyArr(
        order, (pixloc[0] - 2043.5) / 2044.0, (pixloc[1] - 2043.5) / 2044.0
    )
    this_psf = (
        smooth_and_pad(
            np.einsum("a,aij->ij", lpoly, inpsf_cube), tophatwidth=in_psf_oversam
        )
        / 64
    )
    return this_psf


print("returned this_psf and read LPolyArr function!")
sys.stdout.flush()

# More debugging prints
# print(os.getenv('SLURM_NTASKS'))
# print(os.getenv('SLURM_CPUS_PER_TASK'))
ncpu = int(os.getenv("SLURM_NTASKS"))
print(ncpu)
sys.stdout.flush()


def read_config(config_file):
    """Read configuration from YAML file"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


print("returned config and read read_config function!")
sys.stdout.flush()


def read_catalog(file_path, file_type=None):
    """
    Read a star catalog and return RA, Dec, and H-band magnitude.
    """

    # Determining file type
    if file_type is None:
        if file_path.lower().endswith(".fits"):
            file_type = "fits"
        else:
            file_type = "ascii"

    # FITS catalogs (Besancon)
    if file_type == "fits":
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            ra = data["RAJ2000"]
            dec = data["DECJ2000"]
            mag_H = data["H"]

    # ASCII catalogs (Anderson)
    elif file_type == "ascii":
        with open(file_path, "r") as f:
            for line in f:
                if "RA" in line and "Dec" in line:
                    # Remove the '#' and split the line into a list of names
                    cols = line.replace("#", "").split()
                    break

        df = pd.read_table(file_path, sep=r"\s+", comment="#", names=cols)

        ra = df["RA"]
        dec = df["Dec"]

        # Using Anderson H-band column
        mag_H = df["m160_u"] + 1.39 # This is converting from Vega to AB 

    else:
        raise ValueError("Unsupported file type")

    return {"ra": ra, "dec": dec, "mag_H": mag_H}


def makeParser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description="Star Simulation Configuration")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    return parser


print("read makeParser and returned parser!")
sys.stdout.flush()


def sedBB(w, T):
    """Return blackbody flux density at wavelength and temperature."""
    return (
        (8 * np.pi * const.h * const.c**2 / w**5)
        * 1
        / (np.exp(const.h * const.c / (w * const.k_B * T)) - 1)
    ).decompose()


print("retuned value and read sedBB function!")
sys.stdout.flush()


def convert_pos(ra, dec, wcs):
    """Convert RA/Dec to pixel coordinates using the World Coordinate System (WCS)."""
    worldCenter = galsim.CelestialCoord(ra=ra, dec=dec)
    imageCenter = wcs.posToImage(worldCenter)
    return (imageCenter.x, imageCenter.y)


print("returned image center x, y!")
sys.stdout.flush()


def assign_star(x, y):
    """Assigning row of 8x4 processes to draw out stars"""
    # x_blue = np.clip(x // (nside // process_h), min = 0, max = 4088)
    # y_blue = np.clip(y // (nside // process_h), min = 0, max = 4088)
    x_blue_idx = int(np.clip(x // (nside // process_h), 0, process_h - 1))
    y_blue_idx = int(np.clip(y // (nside // process_v), 0, process_v - 1))
    task = (
        y_blue_idx * process_h + x_blue_idx
    )  # fixed this section, will see if this works
    return task


print("returned task for assign_star!")
sys.stdout.flush()


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


print("read j_location and returned process_bounds!")
sys.stdout.flush()


def Eqn(matrix, soln):
    """"""
    return matrix**-1 * soln

    # def OffsetPixel([pixeloffsets], xgrid, ygrid):
    """mat = expectVal(x, 1, 0)
       solveEqn(mat, ..., pixeloffsets)
       gives a, b, c, d, e, f
       compute and return wi: the weights
       also try to return expectation values later
       utilizing function above
       may need to write another function here"""


# Delete functions above later, this needs to be done in separate .py file


def draw_stars(j, cat, wcs, scaNum, nobj, is_in_circle, task_array, effAreaTable, transmissionCurve, tExp, roman_bandpasses, big_fft_params, x_padding=std_pad, y_padding=std_pad):  # Sorry, my code said all of these were undefined, I'll remove this after                       
    """Draw stars for tile index j into a temporary image section."""
    with fits.open(
        "/users/PAS2340/karadiludovico/fits_files/psf_poly.fits"
    ) as inpsf_file:
        psf_data = np.copy(inpsf_file[scaNum].data[:, :, :])
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
            new_image_center = transformPos(imageCenter.x, imageCenter.y)
            imageCenter2 = galsim.PositionD(
                x=new_image_center[0], y=new_image_center[1]
            )

            # Next, using position to compute PSF, use del command
            this_psf = compute_poly(
                psf_data, (new_image_center[0], new_image_center[1])
            )
            psf = galsim.Image(this_psf)
            # psf = galsim.roman.getPSF(scaNum, 'H158', SCA_pos=pos_SCA, wcs=mywcs, wavelength=roman_bandpasses['H158'])
            interp_psf = galsim.InterpolatedImage(
                psf, x_interpolant="lanczos32", scale=1
            )  # 0.11/in_psf_oversam)

            # Rest of flux calculations
            wav = np.arange(0.400, 2.600, 0.001) * u.um
            fluxUnnorm = sedBB(wav, 5000 * u.K)
            fLambdaRef = fNuRef * const.c / wav**2
            mag = cat["mag_H"][i]
            norm = (
                10 ** (-0.4 * mag)
                * np.trapezoid(fLambdaRef * transmissionCurve * wav, x=wav)
                / np.trapezoid(fluxUnnorm * transmissionCurve * wav, x=wav)
            )
            flux = norm * fluxUnnorm
            nPhotQ = np.trapezoid(
                flux * effAreaTable["F158"] * u.m**2 * wav * tExp / (const.h * const.c),
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
            source.drawImage(
                tempImage, method="no_pixel", center=imageCenter2, add_to_image=True
            )
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


print("read through draw_stars function!")
sys.stdout.flush()

# Main Execution
if __name__ == "__main__":
    # Parse command line to get config file path
    parser = makeParser()
    args = parser.parse_args()
    print("read parser!")
    sys.stdout.flush()

    # Read configuration from YAML file
    config = read_config(args.config_file)
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
    wcsFileName = "/users/PCON0003/cond0007/PSF-TEST-FILES/Roman_WAS_simple_model_H158_13814_14.fits"
    readImage = galsim.fits.read(file_name=wcsFileName, hdu=1, read_header=True)
    mybounds = readImage.bounds
    readImage.header["CRVAL1"] = float(config["raCen"])
    readImage.header["CRVAL2"] = float(config["decCen"])
    readImage.header["LONPOLE"] = float(config["LONPOLE"])
    mywcs, neworigin = galsim.wcs.readFromFitsHeader(readImage.header)
    print("read from degrees to mywcs, neworigin!")
    sys.stdout.flush()

    # Determine which stars are in the circle
    """if not config["randomPos"]:
        ra = cat["ra"] * np.pi / 180
        dec = cat["dec"] * np.pi / 180

        world_pos = mywcs.toWorld(galsim.PositionD(x=2044, y=2044))
        cos_theta = np.sin(world_pos.rad[1]) * np.sin(dec) + np.cos(
            world_pos.rad[1]
        ) * np.cos(dec) * np.cos(ra - world_pos.rad[0])
        is_in_circle = cos_theta > np.cos(
            (0.11 * 4088) / (np.sqrt(2) * 3600) * np.pi / 180
        )
        print("read if not config randomPos for with fits.open starCat!")
        sys.stdout.flush()
    else:
        is_in_circle = np.ones(nobj, dtype=bool)
        print("read else statement for is_in_circle!")
        sys.stdout.flush() """
    
    # Determine which stars are in circle
    is_in_circle = np.zeros(nobj, dtype=bool)

    if not config["randomPos"]:
        for i in range(nobj):
            ra = cat["ra"][i] * degrees
            dec = cat["dec"][i] * degrees
            world_pos = galsim.CelestialCoord(ra=ra, dec=dec)
            image_pos = mywcs.posToImage(world_pos)
            x = image_pos.x
            y = image_pos.y

            # Keep stars that fall on the detector
            if (0 <= x < 4088) and (0 <= y < 4088):
                is_in_circle[i] = True
    else:
        is_in_circle[:] = True 
    print("read if not config randomPos!")
    sys.stdout.flush()         
    # Delete this section of code later if this doesn't work      


    # Telescope exposure/SCA
    """scaNum = int(config["SCA"])
    if scaNum < 10:
        effAreaTable = aio.ascii.read(
            "Roman_effarea_tables_20240327/Roman_effarea_v8_SCA0{}_20240301.ecsv".format(
                scaNum
            )
        )
        print("read if scaNum!")
        sys.stdout.flush()
    else:
        effAreaTable = aio.ascii.read(
            "Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{}_20240301.ecsv".format(
                scaNum
            )
        )
        print("read else scaNum!")
        sys.stdout.flush()"""

    # Telescope exposure/SCA
    scaNum = int(config["SCA"])
    effAreaTable = aio.ascii.read(
            f"Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{scaNum:02d}_20240301.ecsv"
            )

    mirrorDiameter = 2.37 * u.m
    geomArea = np.pi * mirrorDiameter**2 / 4
    transmissionCurve = effAreaTable["F158"] * u.m**2 / geomArea
    tExp = 120 * u.s
    print("read from mirrorDiameter to tExp!")
    sys.stdout.flush()

    # Create output image with bounds
    xmin = ymin = -std_pad
    xmax = ymax = nside * in_psf_oversam + std_pad - 1
    outImage = galsim.Image(galsim.BoundsI(xmin, xmax, ymin, ymax))
    print("read outImage!")
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
    multiprocess_stars = functools.partial(draw_stars, cat=cat, wcs=mywcs, scaNum=scaNum, nobj=nobj, is_in_circle=is_in_circle, task_array=task_array, effAreaTable=effAreaTable, transmissionCurve=transmissionCurve, tExp=tExp, roman_bandpasses=roman_bandpasses, big_fft_params=big_fft_params, x_padding=std_pad, y_padding=std_pad)
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
                outImage[result.bounds] += (
                    result  # Combine incrementally, having error with this line, changed it though
                )
                print("read outImage += result!")
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

    # outImage = process_func(0)
    outImage.write(config["outFile"])
    print("Image written to", config["outFile"])
    sys.stdout.flush()
