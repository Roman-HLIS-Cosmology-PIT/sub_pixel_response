import galsim
import galsim.roman
import numpy as np
from astropy import units as u
from astropy.io import fits
import astropy.io as aio
from sub_pixel_response.imagesim import (
    assign_star,
    compute_poly,
    convert_pos,
    j_location,
    l_poly_array,
    sed_bb,
    smooth_and_pad,
    transform_pos,
    draw_stars,
    GLB_DATA,
    GlobalContext
)

# Important constants that are needed to run these unit tests
in_psf_oversam = GLB_DATA["in_psf_oversam"]
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
    Test that smooth_and_pad returns a larger array from added padding in the correct shape.
    """
    input_array = np.ones((4, 4))
    output = smooth_and_pad(input_array, tophatwidth=0)
    npad = int(np.ceil(in_psf_oversam + 2))
    npad += (4 - npad) % 4
    expected_shape = (
        input_array.shape[0] + 2 * npad,
        input_array.shape[1] + 2 * npad,
    )

    assert output.shape[0] > input_array.shape[0]
    assert output.shape[1] > input_array.shape[1]
    assert np.all(np.isfinite(output))
    assert output.shape == expected_shape


def test_l_poly_array():
    """
    Test l_poly_array for a polynomial of order 0 and order 2with known results.
    """
    output = l_poly_array(0, 0.5, 0.5)
    expected = np.array([1.0])
    assert np.allclose(output, expected)

    output = l_poly_array(2, 1, -1)
    expected = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    assert np.allclose(output, expected)


def test_compute_poly():
    """
    Test compute_poly returns a finite, 2D, padded PSF array of the correct shape.
    """
    inpsf_cube = np.ones((1, 4, 4))
    pix = (2044, 2044)
    output = compute_poly(inpsf_cube, pix, order=0)
    expected = (
        smooth_and_pad(
            np.ones((4, 4)),
            tophatwidth=in_psf_oversam,
        )
        / 64
    )

    assert output.ndim == 2
    assert np.all(np.isfinite(output))
    assert output.shape[0] > 4
    assert output.shape[1] > 4
    assert output.shape == expected.shape
    assert np.allclose(output, expected)


def test_sed_bb():
    """Tests the blackbody flux density at wavelength and temperature."""
    wav = np.arange(0.400, 2.600, 0.001) * u.um
    output = sed_bb(wav, 5000 * u.K)
    assert len(output) == len(wav)


"""Tests the conversion of RA/Dec to pixel coordinates using the World Coordinate System (WCS)."""
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


def test_convert_pos():
    """Test for convert position"""
    myheader = fits.Header.fromstring(
        WCSSTRING,
        sep="\n",
    )
    mywcs = galsim.AstropyWCS(header=myheader)
    # Testing for random part of top edge of image first
    ra_1 = 10.2124400 * galsim.degrees
    dec_1 = -44.2785210 * galsim.degrees
    x_1, y_1 = convert_pos(ra_1, dec_1, mywcs)
    expected_x_1 = 2035.877414376633
    expected_y_1 = 4087.1269146931413
    assert np.allclose(x_1, expected_x_1, atol=1)
    assert np.allclose(y_1, expected_y_1, atol=1)
    # Now testing for random part of right edge of image
    ra_2 = 10.2950082 * galsim.degrees
    dec_2 = -44.3375139 * galsim.degrees
    x_2, y_2 = convert_pos(ra_2, dec_2, mywcs)
    expected_x_2 = 4085.879172318449
    expected_y_2 = 2135.1264219908003
    assert np.allclose(x_2, expected_x_2, atol=1)
    assert np.allclose(y_2, expected_y_2, atol=1)


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

def test_draw_stars():
    """Test that draw_stars works"""
    with GlobalContext({"nside":128}):
        j = 0 # need to fix
        cat = {"RA":np.array([]), "DEC":np.array([]), "MAG":, "is_in_circle":} # need to fix
        wcs = 0 # need to fix
        sca_num = 7
        task_array = 0 # need to fix
        eff_area_table = aio.ascii.read(
        f"Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{sca_num:02d}_20240301.ecsv"
    )
        t_exp = 100
        roman_bandpasses = galsim.roman.getBandpasses()
        big_fft_params = galsim.GSParams(maximum_fft_size=123000)
        psf_file = "tests/data/psf_poly_14only.fits"
        filter_name = "F158"
        image = draw_stars(j, cat, wcs, sca_num, task_array, eff_area_table, t_exp, roman_bandpasses, big_fft_params, psf_file, filter_name)
        assert image.shape == (128, 128)
