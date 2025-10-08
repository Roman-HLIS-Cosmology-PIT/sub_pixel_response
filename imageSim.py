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
import datetime
import pytz

# in SBATCH file, added mem=100G in there 
# if it doesn't work or make a difference, take out later
# uncomment sys.
# request more time, but should be taking less time to execute the code

"""

- Don't think this file is right, but I fixed everything that was undefined lol
- Need to figure out how to rewrite process_stars into the draw_stars function
- Also need to learn how to connect draw_stars function with j_location and assign_star 
  functions in order to draw the stars to their correct assigned process
- More with this later

"""

def print_report(s):
    print(s, datetime.datetime.now(pytz.timezone("UTC")).strftime("%Y%m%d%H%M%S%f")) # favorite format for time
    sys.stdout.flush()

# Global data
in_psf_oversam = 6
fNuRef = 3.631e-23*(u.W/u.m**2)/u.Hz  # W/m^2/Hz
process_h = 4
process_v = 8
nside = 4088
std_pad = 24
print('read global data!') 
sys.stdout.flush()

def transformPos(x, y, oversam=6):
    X = oversam * (x - 0.5) + 0.5
    Y = oversam * (y - 0.5) + 0.5
    return (X, Y)
print('transformPos read!')
sys.stdout.flush()

def smooth_and_pad(inArray: np.array, tophatwidth: float = 0.0, 
                       gaussiansigma: float = 0.0) -> np.array:
        '''
        Utility to smear a PSF with a tophat and a Gaussian.

        Parameters
        ----------
        inArray : np.array, shape : (ny, nx)
            Input PSF array to be smeared.
        tophatwidth : float, optional
        gaussiansigma : float, optional
            Both in units of the pixels given (not native pixel). The default is 0.0.

        Returns
        -------
        outArray : np.array, shape : (ny+npad*2, nx+npad*2)
            Smeared input PSF array.

        '''

        npad = int(np.ceil(tophatwidth + 6*gaussiansigma + 1))
        npad += (4-npad) % 4  # make a multiple of 4
        (ny, nx) = np.shape(inArray)
        nyy = ny+npad*2
        nxx = nx+npad*2
        outArray = np.zeros((nyy, nxx))
        outArray[npad:-npad, npad:-npad] = inArray
        outArrayFT = np.fft.fft2(outArray) 

        # convolution (with Fourier Transform)
        uy = np.linspace(0, nyy-1, nyy)/nyy
        uy = np.where(uy > .5, uy-1, uy)
        ux = np.linspace(0, nxx-1, nxx)/nxx
        ux = np.where(ux > .5, ux-1, ux) 
        outArrayFT *= np.sinc(ux[None, :]*tophatwidth)*np.sinc(uy[:, None]*tophatwidth) * \
            np.exp(-2.*np.pi**2*gaussiansigma ** 2*(ux[None, :]**2+uy[:, None]**2))

        outArray = np.real(np.fft.ifft2(outArrayFT))
        return outArray 
print('returned outArray and read smooth_and_pad function!')
sys.stdout.flush()

def LPolyArr(PORDER,u_,v_):
    '''
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
    '''

    ua = np.ones(PORDER+1)
    va = np.ones(PORDER+1)
    for m in range(1,PORDER+1):
        L = legendre(m)
        ua[m] = L(u_)
        va[m] = L(v_)
    arr = np.outer(va,ua).flatten()
    return arr 
print('returned LPolyArr!')
sys.stdout.flush()

def compute_poly(inpsf_cube, pixloc, order=1):
    lpoly = LPolyArr(order, (pixloc[0]-2043.5)/2044., (pixloc[1]-2043.5)/2044.)    
    this_psf = smooth_and_pad(np.einsum('a,aij->ij', lpoly, inpsf_cube), tophatwidth=in_psf_oversam)/64

    return this_psf
print('returned this_psf and read LPolyArr function!')
sys.stdout.flush()

#print(os.getenv('SLURM_NTASKS'))
#print(os.getenv('SLURM_CPUS_PER_TASK'))
ncpu = int(os.getenv('SLURM_NTASKS'))
print(ncpu)
sys.stdout.flush()

def read_config(config_file):
    """Read configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config 
print('returned config and read read_config function!')
sys.stdout.flush()

def makeParser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description="Star Simulation Configuration")
    parser.add_argument('config_file', help='Path to YAML configuration file')
    return parser
print('read makeParser and returned parser!')
sys.stdout.flush()

def sedBB(w, T):
    return ((8*np.pi*const.h*const.c**2/w**5)*1/(np.exp(const.h*const.c/(w*const.k_B*T))-1)).decompose()
print('retuned value and read sedBB function!')
sys.stdout.flush()

def convert_pos(ra, dec, wcs):
    worldCenter = galsim.CelestialCoord(ra=ra, dec=dec) 
    imageCenter = wcs.posToImage(worldCenter)
    return (imageCenter.x, imageCenter.y)
print('returned image center x, y!')
sys.stdout.flush()

def assign_star(x, y):
    """Assigning row of 8x4 processes to draw out srars"""
    x_blue = np.clip(x // (nside // process_h), min = 0, max = 4088)
    y_blue = np.clip(y // (nside // process_h), min = 0, max = 4088)
    task = y_blue * process_h + x_blue
    return task
print('returned task for assign_star!')
sys.stdout.flush()

# j for given process number
def j_location(process, x_padding=0, y_padding=0):
    # padding is in subpixel coordinates
    xmin_j = (nside // process_h * (process % process_h)) * in_psf_oversam
    ymin_j = (nside // process_v * (process // process_h)) * in_psf_oversam
    xmax_j = (in_psf_oversam * nside // process_h) + xmin_j - 1
    ymax_j = (in_psf_oversam * nside // process_v) + ymin_j - 1 
    process_bounds = galsim.BoundsI(xmin_j - x_padding, xmax_j + x_padding, ymin_j - y_padding, ymax_j + y_padding)
    return process_bounds 
print('read j_location and returned process_bounds!')
sys.stdout.flush()

# draw_stars function was fixed and this looks good! :)
# added try statement, doesn't seem to be working for some reason 
def draw_stars(j, cat, wcs):
    with fits.open('/users/PAS2340/karadiludovico/psf_poly.fits') as inpsf_file:
        psf_data = np.copy(inpsf_file[scaNum].data[:, :, :]) 
    try:
        mybounds = j_location(j, x_padding=std_pad, y_padding=std_pad) 
        tempImage = galsim.Image(bounds=mybounds, dtype=np.float32)
        for i in range(10000):
            if task_array[i] != j:
                continue
            if not is_in_circle[i]:
                continue
            if i != 9949:
                continue
            
            # First, calculating position
            degrees = galsim.AngleUnit(np.pi/180)
            ra = cat.get(i, 'RAJ2000') * degrees
            dec = cat.get(i, 'DECJ2000') * degrees
            worldCenter = galsim.CelestialCoord(ra=ra, dec=dec) 
            imageCenter = wcs.posToImage(worldCenter)
            new_image_center = transformPos(imageCenter.x, imageCenter.y)
            imageCenter2 = galsim.PositionD(x = new_image_center[0], y = new_image_center[1])

            # Next, using position to compute PSF, use del command 
            this_psf = compute_poly(psf_data, (new_image_center[0], new_image_center[1]))
            psf = galsim.Image(this_psf) 
            # psf = galsim.roman.getPSF(scaNum, 'H158', SCA_pos=pos_SCA, wcs=mywcs, wavelength=roman_bandpasses['H158'])
            interp_psf = galsim.InterpolatedImage(psf, x_interpolant = 'lanczos32', scale=1)#0.11/in_psf_oversam) 

            # Rest of flux calculations
            wav = np.arange(0.400, 2.600, 0.001) * u.um
            fluxUnnorm = sedBB(wav, 5000*u.K)
            fLambdaRef = fNuRef * const.c / wav**2
            mag = cat.get(i, 'H')
            norm = 10**(-0.4*mag) * np.trapezoid(fLambdaRef*transmissionCurve*wav, x=wav)/np.trapezoid(fluxUnnorm*transmissionCurve*wav, x=wav)
            flux = norm*fluxUnnorm
            nPhotQ = np.trapezoid(flux*effAreaTable['F158']*u.m**2*wav*tExp/(const.h * const.c), x=wav)
            nPhotQ = nPhotQ.decompose()
            nPhot = nPhotQ.value 
            if not np.isfinite(nPhot):
                print(f"!! WARNING (j={j}, i={i}): Invalid flux calculated: {nPhot}", flush=True)
                continue # Skip this star
            if not mybounds.includes(imageCenter2):
                print(f"!! WARNING (j={j}, i={i}): Star position {imageCenter2} is outside bounds {mybounds}", flush=True)
                continue # Skip this star
            st_model = galsim.DeltaFunction(flux=nPhot)
            source = galsim.Convolve([interp_psf,st_model], gsparams=big_fft_params) 
            print('read flux calculations per star!', i, j)
            sys.stdout.flush()
            source.drawImage(tempImage, method='no_pixel', center=imageCenter2, add_to_image = True) 
            print('read source.drawImage!')
            sys.stdout.flush()
            del psf, this_psf, interp_psf 
        return tempImage
    except Exception as e:
        print(f"Error in process {j}: {str(e)}", file=sys.stderr)    
        raise 
    print(j_location)
    print('read draw_stars function and returned tempImage!')
    sys.stdout.flush()
print('read through draw_stars function!')
sys.stdout.flush()

# need to figure out how to write config yaml stuff for randomPos and blackBody later on 
# once this code works and executes well lol

# don't need this function to use for the code anymore, didn't call it anywhere else    
"""     
def process_stars(j, nstar, config, cat, is_in_circle, mywcs, mybounds, scaNum, transmissionCurve, 
                roman_bandpasses, tExp, geomArea):
    Process single star (to be run in parallel)
    inpsf_file = fits.open('/users/PAS2340/karadiludovico/psf_poly.fits') 
    temp_image = galsim.Image(bounds=mybounds, wcs=mywcs) 
    temp_image2 = galsim.Image(ncol=24528, nrow=24528, scale=1) 
    ncpu = int(os.getenv('SLURM_NTASKS')) 
    print('read inpsf_file, temp images, and ncpu')
    # sys.stdout.flush()
    for i in range(j, nstar, ncpu): # replacing this section, change location of it? 
        # check if star is in bounds
        if not is_in_circle[i]:
            continue 
        print('read if statement for i in range and continued!')
        # sys.stdout.flush()
        # Calculate flux
        if config['blackBody']:
            mag = cat.get(i, 'H')
            wav = np.arange(0.400, 2.600, 0.001) * u.um
            fluxUnnorm = sedBB(wav, 5000*u.K)
            fLambdaRef = fNuRef * const.c / wav**2
            norm = 10**(-0.4*mag) * np.trapezoid(fLambdaRef*transmissionCurve*wav, x=wav)/np.trapezoid(fluxUnnorm*transmissionCurve*wav, x=wav)
            flux = norm*fluxUnnorm
            nPhotQ = np.trapezoid(flux*effAreaTable['F158']*u.m**2*wav*tExp/(const.h * const.c), x=wav)
            nPhotQ = nPhotQ.decompose()
            nPhot = nPhotQ.value 
            print('read if statement for if config blackBody!')
            # sys.stdout.flush()    
        else:
            mag = cat.get(i, 'H')
            nPhot = 300000  # Need to update as a function of SED for a given star
            # nPhot is placeholder and should be calculated properly
            print('read else statement for config blackBody!')
            # sys.stdout.flush()

        # Calculate position
        if not config['randomPos']:
            degrees = galsim.AngleUnit(np.pi/180)
            ra = cat.get(i, 'RAJ2000')*degrees
            dec = cat.get(i, 'DECJ2000')*degrees
            print('read not config randomPos!')
            # sys.stdout.flush()

            wcs = mywcs
            worldCenter = galsim.CelestialCoord(ra=ra, dec=dec) 
            imageCenter = wcs.posToImage(worldCenter)
            
            if (imageCenter.x > mybounds.getXMax() or 
                imageCenter.y > mybounds.getYMax() or 
                imageCenter.x < mybounds.getXMin() or imageCenter.y < mybounds.getYMin()):
                continue  # Skip if out of bounds
            print('read if statement for image center and continued!')
            # sys.stdout.flush()
            new_image_center = transformPos(imageCenter.x, imageCenter.y)
            imageCenter2 = galsim.PositionD(x = new_image_center[0], y = new_image_center[1])
            print('read new image center and image center 2!')
            # sys.stdout.flush()
            # pos_SCA = galsim.PositionD(x=imageCenter.x-(mybounds.getXMax()/2.), y=imageCenter.y-(mybounds.getYMax()/2.))
            psf_data = inpsf_file[scaNum].data[:, :, :] 
            this_psf = compute_poly(psf_data, (new_image_center[0], new_image_center[1])) 
            psf = galsim.Image(this_psf) 
            print('read psf_data, this_psf, and psf!')
            # sys.stdout.flush()
            # psf = galsim.roman.getPSF(scaNum, 'H158', SCA_pos=pos_SCA, wcs=mywcs, wavelength=roman_bandpasses['H158'])
            interp_psf = galsim.InterpolatedImage(psf, x_interpolant = 'lanczos32', scale=1)#0.11/in_psf_oversam) 
            st_model = galsim.DeltaFunction(flux=nPhot)
            source = galsim.Convolve([interp_psf,st_model], gsparams=big_fft_params) 
            print('read interp_psf, st_model, and source!')
            # sys.stdout.flush()
        else:
            x = np.random.random_sample()*mybounds.getXMax()
            y = np.random.random_sample()*mybounds.getYMax()
            imageCenter2 = galsim.PositionD(x=x, y=y)
            pos_SCA = galsim.PositionD(x=x-(mybounds.getXMax()/2.), y=y-(mybounds.getYMax()/2.))
            psf = galsim.roman.getPSF(scaNum, 'H158', SCA_pos=pos_SCA, wcs=mywcs, wavelength=roman_bandpasses['H158'])
            source = psf*nPhot
            print('read else statement for not config randomPos!')
            # sys.stdout.flush()

        source.drawImage(temp_image2, method='no_pixel', center=imageCenter2, add_to_image = True) 
        print("Star Drawn!", i)
        print("Process Complete!", j)
    return temp_image2
    print('returned temp_image 2 sucessfully!')
    # sys.stdout.flush()   
"""   

if __name__ == '__main__':
    # Parse command line to get config file path
    parser = makeParser()
    args = parser.parse_args()
    print('read parser!')
    sys.stdout.flush()
    
    # Read configuration from YAML file
    config = read_config(args.config_file)
    print('read config!')
    sys.stdout.flush()
    
    # Read RA,Dec from star catalog
    try:
        assert('.fits' in config['starCat'])
    except:
        raise Exception("Star Catalog should be a .fits file")
    print('executed try, assert, except, and raise for reading ra and dec from star catalog!')
    sys.stdout.flush()

    cat = galsim.Catalog(config['starCat'])
    #cat = cat[:1000] # added this line to only print out first 1000 stars in image
    print('read cat!')
    sys.stdout.flush()
    
    degrees = galsim.AngleUnit(np.pi/180)
    wcsFileName = '/users/PCON0003/cond0007/PSF-TEST-FILES/Roman_WAS_simple_model_H158_13814_14.fits'
    readImage = galsim.fits.read(file_name=wcsFileName, hdu=1, read_header=True)
    mybounds = readImage.bounds
    readImage.header['CRVAL1'] = float(config['raCen'])
    readImage.header['CRVAL2'] = float(config['decCen'])
    mywcs, neworigin = galsim.wcs.readFromFitsHeader(readImage.header)
    print('read from degrees to mywcs, neworigin!')
    sys.stdout.flush()

    # Determine which stars are in the circle
    if not config["randomPos"]:
        with fits.open(config["starCat"]) as f:
            ra = f[1].data["RAJ2000"] * np.pi / 180
            dec = f[1].data["DECJ2000"] * np.pi / 180
            world_pos = mywcs.toWorld(galsim.PositionD(x=2044, y=2044))
            cos_theta = np.sin(world_pos.rad[1]) * np.sin(dec) + np.cos(world_pos.rad[1]) * np.cos(dec) * np.cos(ra - world_pos.rad[0])
            is_in_circle = cos_theta > np.cos((0.11 * 4088) / (np.sqrt(2) * 3600) * np.pi / 180)
            print('read if not config randomPos for with fits.open starCat!')
            sys.stdout.flush()
    else:
        is_in_circle = np.ones(cat.nobjects, dtype=bool) 
        print('read else statement for is_in_circle!')
        sys.stdout.flush()
               
    scaNum = int(config['SCA'])
    if scaNum < 10:
        effAreaTable = aio.ascii.read('Roman_effarea_tables_20240327/Roman_effarea_v8_SCA0{}_20240301.ecsv'.format(scaNum))
        print('read if scaNum!')
        sys.stdout.flush()
    else:
        effAreaTable = aio.ascii.read('Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{}_20240301.ecsv'.format(scaNum))
        print('read else scaNum!')
        sys.stdout.flush()

    mirrorDiameter = 2.37*u.m
    geomArea = np.pi*mirrorDiameter**2/4
    transmissionCurve = effAreaTable['F158']*u.m**2/geomArea
    tExp = 120*u.s
    print('read from mirrorDiameter to tExp!')
    sys.stdout.flush()

    # Create output image with bounds
    xmin = ymin = -std_pad 
    xmax = ymax = nside * in_psf_oversam + std_pad - 1
    outImage = galsim.Image(galsim.BoundsI(xmin, xmax, ymin, ymax)) 
    print('read outImage!')
    sys.stdout.flush()
    
    roman_bandpasses = galsim.roman.getBandpasses()
    big_fft_params = galsim.GSParams(maximum_fft_size=123000)
    print('read roman_bandpasses and big_fft_params!')
    sys.stdout.flush()

    task_array = np.zeros(cat.nobjects, dtype=np.int32)

    for i in range(cat.nobjects):
        degrees = galsim.AngleUnit(np.pi/180)
        ra = cat.get(i, 'RAJ2000')*degrees
        dec = cat.get(i, 'DECJ2000')*degrees
        x, y = convert_pos(ra, dec, mywcs)
        j = assign_star(x, y) 
        task_array[i] = j # changed this line from cat.nobjects[i] = j
        # at line above, there was a mistake where 'int' object does not support item assignemnt
    
    # Prepare arguments for parallel processing
    multiprocess_stars = functools.partial(draw_stars, 
        cat=cat, 
        wcs=mywcs)
    print('read multiprocess_stars!') 
    sys.stdout.flush()

    # Determine number of processes to use
    num_processes = min(ncpu, cat.nobjects)

    print(num_processes)
    sys.stdout.flush()
    
    # Parallel processing and combine results
    with Pool(processes=num_processes) as pool:
        print('read with Pool statement!')
        sys.stdout.flush() 
        for result in pool.imap_unordered(multiprocess_stars, range(num_processes)):
            print('read for result in pool statement!')
            sys.stdout.flush() 
            if result is not None:
                print('read if results is not None!')
                sys.stdout.flush()
                outImage[result.bounds] += result # Combine incrementally, having error with this line, changed it though
                print('read outImage += result!')
                sys.stdout.flush()
        print('ran pool for parallel processing!')
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
    outImage.write(config['outFile'])
    print("Image written to", config['outFile']) 
    sys.stdout.flush()