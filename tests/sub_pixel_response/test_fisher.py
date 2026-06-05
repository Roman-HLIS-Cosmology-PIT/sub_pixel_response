import numpy as np
import pytest
from sub_pixel_response.fisher import fisher

# this is a corner of Jay Anderson's file, used here just for format testing
EXAMPLE_ANDERSON = """#
# COL01 --    x2017: the x position at the 2017.38 epoch
# COL02 --    y2017: the y position at the 2017.38 epoch
# COL03 --     m606: the calibrated F606W vegamag photometry in VEGAMAGS
# COL04 --   m160_u: the "best" H-band vega magnitude (from F160W HST or interpolation or isochrone)
# COL05 --   jmag_u: the "best" J-band vega magnitude (from HAWK-I or interpolation or isochrone)
# COL06 --   kmag_u: the "best" K-band vega magnitude (from HAWK-I or interpolation or isochrone)
#
  5495.022   649.326 24.2655 22.7555 23.0829 22.6821
  5512.111   653.009 22.4738 21.4022 21.6398 21.3838
  5498.976   657.033 23.3841 22.0271 22.3219 21.9785
  5509.040   657.877 22.0525 21.1132 21.3218 21.1047
  5535.178   658.591 22.9901 21.7629 22.0325 21.7294
  5542.126   660.244 24.5785 22.9534 23.2997 22.8621
  5498.702   669.159 23.4768 22.1917 22.4810 22.1522
  5469.278   673.576 22.0075 21.0838 21.2891 21.0762
  5530.086   673.454 22.4002 21.3198 21.5665 21.3016
  5460.965   674.532 24.8899 22.8754 23.3061 22.7550
"""


def test_fisher(tmp_path):
    """Simple test function for Fisher."""

    # write the input file
    with open(str(tmp_path) + "/example.txt", "w") as file:
        file.write(EXAMPLE_ANDERSON)

    cat = fisher.StarCat(str(tmp_path) + "/example.txt", seed=42)
    cat.area /= 1e4  # make a lot of copies!
    assert cat.nstar == 10
    err = cat.mags[7] - np.array([22.108, 22.199, 22.474, 22.966])
    assert np.all(np.abs(err) < 0.02)

    N = 20
    bins = np.logspace(1, 3, N + 1)
    hist, cr = cat.get_cr_function(158, bins)
    assert 35.4 < cr[5] < 35.6
    assert 2.1e-3 < hist[5] < 2.2e-3
    assert hist[7] < 1e-6

    # parameters for this test
    b = 158
    fracpix = 0.75

    ivar_mean, ivar_std = cat.fisher_properties(b, 5.0, 1000.0, 63.25, 2900)
    m1 = ivar_mean * fracpix
    s1 = ivar_std * np.sqrt(fracpix)
    ivars, sigma_epsf = cat.fisher_realizations(b, 5.0, 1000.0, 63.25, 2900, fracpix, 50000)
    m2 = np.mean(ivars)
    s2 = np.std(ivars)
    assert 0.99 < m1 / m2 < 1.01
    assert 0.99 < s1 / s2 < 1.01
    assert 1.02 < sigma_epsf * m1**0.5 < 1.05

    # now check that the "official" Fisher runs
    # due to the small catalog size, this will raise a warning
    with pytest.warns(RuntimeWarning):
        fisher.fisher(str(tmp_path) + "/example.txt")
