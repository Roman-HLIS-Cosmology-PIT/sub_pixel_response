# Tests for four out of eight functions for imageSim.py


# Imports will be placed here
import galsim
import galsim.roman
import numpy as np
from astropy import units as u
from sub_pixel_response.imageSim import assign_star, convert_pos, j_location, sed_bb

# Important constants that are needed to run these unit tests
# Not sure if these should be inside the functions??
nside = 4088
process_h = 4
process_v = 8
in_psf_oversam = 6
std_pad = 24


def test_sed_bb():
    """Tests the blackbody flux density at wavelength and temperature."""
    wav = np.arange(0.400, 2.600, 0.001) * u.um
    output = sed_bb(wav, 5000 * u.K)
    assert len(output) == len(wav)


# may rewrite this later with np.assert


def test_convert_pos(cat):
    """Tests the conversion of RA/Dec to pixel coordinates using the World Coordinate System (WCS)."""
    degrees = galsim.AngleUnit(np.pi / 180)
    wcs_file_name = "/users/PCON0003/cond0007/PSF-TEST-FILES/Roman_WAS_simple_model_H158_13814_14.fits"
    read_image = galsim.fits.read(file_name=wcs_file_name, hdu=1, read_header=True)
    mywcs = galsim.wcs.readFromFitsHeader(read_image.header)
    ra = cat["ra"][0] * degrees
    dec = cat["dec"][0] * degrees
    x, y = convert_pos(ra, dec, mywcs)
    expected_x = 2048
    expected_y = 2048
    assert np.allclose(x, expected_x, atol=5)
    assert np.allclose(y, expected_y, atol=5)


# Maybe use np.isfinite here??


def test_assign_star():
    """Testing the assigning of the row of 8x4 processes to draw out stars"""
    x = 100
    y = 200
    j = assign_star(x, y)
    expected_result = 0
    assert np.allclose(j, expected_result)


# Going back to this one in a little bit
# add assert line here when done


def test_j_location():
    """Testing function to get tile bounding region in oversampled pixel coordinates."""
    process = 0
    mybounds = j_location(process, x_padding=std_pad, y_padding=std_pad)
    tempImage = galsim.Image(bounds=mybounds, dtype=np.float32)
    assert tempImage.bounds == mybounds
