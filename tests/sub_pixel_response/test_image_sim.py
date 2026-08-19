import io
import urllib.request
from contextlib import redirect_stdout
from importlib.resources import files
from pathlib import Path

import astropy.io as aio
import galsim
import galsim.roman
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from numpy.random import RandomState
from sub_pixel_response import process_image
from sub_pixel_response.imagesim import (
    GLB_DATA,
    GlobalContext,
    assign_star,
    compute_poly,
    convert_pos,
    draw_stars,
    j_location,
    l_poly_array,
    make_final_image,
    print_report,
    run_simulation,
    sed_bb,
    smooth_and_pad,
    transform_pos,
)
from sub_pixel_response.utils.randomutils import get_randpts

# Important constants that are needed to run these unit tests
in_psf_oversam = GLB_DATA["in_psf_oversam"]
std_pad = 24

# Getting the PSF file from the wiki page
PSF_FILE = "https://github.com/Roman-HLIS-Cosmology-PIT/sub_pixel_response/wiki/files/psf_poly_14only.fits.gz"

# Defining the path to the Roman_effarea_tables_20240327 directory before test functions
TEST_DIR = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def get_psf_file(tmp_path_factory):
    """Pull the PSF file."""
    download_dir = tmp_path_factory.mktemp("downloads")
    psf_file = str(download_dir) + "/psf_poly_14only.fits.gz"
    urllib.request.urlretrieve(PSF_FILE, psf_file)
    return psf_file


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


def test_draw_stars(get_psf_file):
    """Test that draw_stars works"""

    psf_file = get_psf_file

    rs = RandomState(22)  # Set a fixed seed

    # Test values for WCS RA and Dec for generating random points
    wcs_ra = 80.5
    wcs_dec = -69.5
    pix_coord1 = 1
    pix_coord2 = 1
    pts_ra, pts_dec = get_randpts(wcs_ra, wcs_dec, 0.002, 400, rng=rs)
    rand_cat = {"ra": pts_ra, "dec": pts_dec, "mag_H": rs.uniform(14, 20, 400)}

    with GlobalContext({"nside": 128, "furry_parakeet": True}):
        j = 0  # need to fix, maybe using this value is fine for the time being
        cat = rand_cat
        myheader = fits.Header.fromstring(
            WCSSTRING,
            sep="\n",
        )
        myheader["CRVAL1"] = wcs_ra
        myheader["CRVAL2"] = wcs_dec
        myheader["CRPIX1"] = pix_coord1
        myheader["CRPIX2"] = pix_coord2
        wcs = galsim.AstropyWCS(header=myheader)
        sca_num = 14
        task_array = np.zeros(400, dtype=np.int32)  # need to fix
        data_dir = files("sub_pixel_response.Roman_effarea_tables_20240327")
        filename = f"Roman_effarea_v8_SCA{sca_num:02d}_20240301.ecsv"
        eff_area_path = data_dir.joinpath(filename)
        eff_area_table = aio.ascii.read(eff_area_path)
        t_exp = 100
        roman_bandpasses = galsim.roman.getBandpasses()
        big_fft_params = galsim.GSParams(maximum_fft_size=123000)
        filter_name = "F158"
        image = draw_stars(
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
        )

        # shape for (1/8, 1/4) of canvas, * 6 oversample + 24 pad on each side
        expect_ny = 128 // 8 * 6 + 2 * 24
        expect_nx = 128 // 4 * 6 + 2 * 24
        assert image.array.shape == (expect_ny, expect_nx)
        # fits.PrimaryHDU(image.array).writeto("a.fits", overwrite=True)
        # np.savetxt("b.txt", np.stack((pts_ra, pts_dec)).T)

        # check there is a star in the right place
        assert np.allclose(
            image.array[63:66, 78:81],
            np.array(
                [
                    [41679.305, 43963.94, 42894.777],
                    [44032.168, 46481.742, 45375.22],
                    [43113.59, 45536.348, 44485.004],
                ]
            ),
            atol=1.0,
            rtol=0.01,
        )
        assert 100 < np.percentile(image.array, 50) < 150
        assert 1000 < np.percentile(image.array, 90) < 2000
        assert 5000 < np.percentile(image.array, 99) < 15000


def test_run_simulation(tmp_path, get_psf_file):
    """Test that draw_stars works"""

    tmp_dir = str(tmp_path)
    psf_file = get_psf_file

    rs = RandomState(22)  # Set a fixed seed

    # Test values for WCS RA and Dec for generating random points
    wcs_ra = 80.5
    wcs_dec = -69.5
    pts_ra, pts_dec = get_randpts(wcs_ra, wcs_dec, 0.1, 2000, rng=rs)
    rand_cat = {"ra": pts_ra, "dec": pts_dec, "mag_H": rs.uniform(14, 20, 2000)}
    # Now put that random catalog in a file.
    star_cat = tmp_dir + "/starcat.fits"
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RAJ2000", format="D", array=rand_cat["ra"]),
            fits.Column(name="DECJ2000", format="D", array=rand_cat["dec"]),
            fits.Column(name="H", format="E", array=rand_cat["mag_H"]),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(star_cat, overwrite=True)
    print("-->", star_cat)

    # Write the YAML
    output_target = tmp_dir + "/testimage.fits"
    with open(tmp_dir + "/config.yaml", "w") as f:
        f.write("---\n")
        f.write("raCen: 80.5\n")
        f.write("decCen: -69.49\n")  # set 0.01 deg from circle center
        f.write(f"starCat: {star_cat}\n")
        f.write(f"PSFFILE: {psf_file}\n")
        f.write("LONPOLE: 225\n")
        f.write("SCA: 14\n")
        f.write("FILTER: F158\n")
        f.write("randomPos: false\n")
        f.write("blackBody: true\n")
        f.write(f"outFile: {output_target}\n")
        f.write("OLDWCS: true\n")
        f.write("...\n")

    with GlobalContext({"nside": 512, "furry_parakeet": True}):
        run_simulation(tmp_dir + "/config.yaml")
    with fits.open(output_target) as f:
        assert np.shape(f[0].data) == (3120, 3120)
        assert 3.5e4 < f[0].data[2742, 387] < 4.0e4  # check there's a star there


def test_make_final_image(tmp_path):
    """Test for filtering and making a final image."""

    # Test setup
    test_ov = 4
    test_ns = 256

    # Make an array of offsets.
    offsets = process_image.generate_offset_array(
        np.array([1.0, 0.0, 0.0, 1.0 / 12.0, 0.0, 1.0 / 12.0]), imageSize=test_ns
    )
    offset_file = str(tmp_path) + "/o.fits"
    fits.PrimaryHDU(offsets).writeto(offset_file, overwrite=True)

    # Error targets
    desired_errs = [0.004, 0.0003, 2e-5, 2e-6]

    # Now check each one.
    for j in range(4):
        # Now make a "test" image that's a single Fourier mode
        u, v = 0.4 / 2**j, 0.2 / 2**j
        nos = test_ov * test_ns
        _s = np.linspace(0, nos - 1, nos) / test_ov - 0.5 * (1 - 1 / test_ov)
        _x, _y = np.meshgrid(_s, _s)
        orig = np.cos(2.0 * np.pi * (u * _x + v * _y)).astype(np.float32)
        del _s, _x, _y

        # Get the down-sampled image
        with GlobalContext({"nside": test_ns, "furry_parakeet": True}):
            final_image = make_final_image(orig, offset_file, oversample=test_ov)

        # Now see what we made
        _s = np.linspace(0, test_ns - 1, test_ns)
        _x, _y = np.meshgrid(_s, _s)
        target = np.cos(2.0 * np.pi * (u * _x + v * _y)) * np.sinc(u) * np.sinc(v) * test_ov**2
        del _s, _x, _y
        err = np.amax(np.abs(target - final_image)) / test_ov**2
        print(u, v, err)

        assert err < desired_errs[j]


def test_report():
    """Test printing message to terminal."""
    f = io.StringIO()
    with redirect_stdout(f):
        print_report("oops")
    assert str(f.getvalue())[:4] == "oops"
