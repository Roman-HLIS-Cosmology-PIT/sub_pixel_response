import galsim
import galsim.roman
import numpy as np
from astropy import units as u
from sub_pixel_response.imagesim import (
    assign_star,
    compute_poly,
    convert_pos,
    j_location,
    l_poly_array,
    sed_bb,
    smooth_and_pad,
    transform_pos,
)

# Important constants that are needed to run these unit tests
nside = 4088
process_h = 4
process_v = 8
in_psf_oversam = 6
std_pad = 24


def test_transform_pos():
    """
    Test the transform_pos function with a simple known input.
    """
    x = 1
    y = 2
    expected_output = (3.5, 9.5)
    output = transform_pos(x, y)
    assert np.allclose(output, expected_output), f"Expected {expected_output}, got {output}"


def test_smooth_and_pad():
    """
    Test that smooth_and_pad returns a larger array from added padding.
    """
    input_array = np.ones((4, 4))

    output = smooth_and_pad(input_array, tophatwidth=0)

    assert output.shape[0] > input_array.shape[0]
    assert output.shape[1] > input_array.shape[1]
    assert np.all(np.isfinite(output))


def test_l_poly_array():
    """
    Test l_poly_array for a polynomial of order 0 with known results.
    """
    output = l_poly_array(0, 0.5, 0.5)

    expected = np.array([1.0])
    assert np.allclose(output, expected)


def test_compute_poly():
    """
    Test compute_poly returns a finite, 2D, padded PSF array.
    """
    inpsf_cube = np.ones((1, 4, 4))
    pix = (2044, 2044)
    output = compute_poly(inpsf_cube, pix, order=0)

    assert output.ndim == 2
    assert np.all(np.isfinite(output))
    assert output.shape[0] > 4
    assert output.shape[1] > 4


def test_sed_bb():
    """Tests the blackbody flux density at wavelength and temperature."""
    wav = np.arange(0.400, 2.600, 0.001) * u.um
    output = sed_bb(wav, 5000 * u.K)
    assert len(output) == len(wav)


def test_convert_pos():
    """Tests the conversion of RA/Dec to pixel coordinates using the World Coordinate System (WCS)."""
    wcs_file_name = "/users/PCON0003/cond0007/PSF-TEST-FILES/Roman_WAS_simple_model_H158_13814_14.fits"
    read_image = galsim.fits.read(file_name=wcs_file_name, hdu=1, read_header=True)
    mywcs, origin = galsim.wcs.readFromFitsHeader(read_image.header)
    ra = 10.208584415642562 * galsim.degrees
    dec = -44.33853770184239 * galsim.degrees
    x, y = convert_pos(ra, dec, mywcs)
    expected_x = 2044.0
    expected_y = 2044.0
    assert np.allclose(x, expected_x, atol=5)
    assert np.allclose(y, expected_y, atol=5)


def test_assign_star():
    """Testing the assigning of the row of 8x4 processes to draw out stars"""
    x = 100
    y = 200
    j = assign_star(x, y)
    expected_result = 0
    assert np.allclose(j, expected_result)


def test_j_location():
    """Testing function to get tile bounding region in oversampled pixel coordinates."""
    process = 0
    mybounds = j_location(process, x_padding=std_pad, y_padding=std_pad)
    tempImage = galsim.Image(bounds=mybounds, dtype=np.float32)
    assert tempImage.bounds == mybounds
