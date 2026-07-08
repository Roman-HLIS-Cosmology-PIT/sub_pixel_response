import numpy as np
from numpy.random import RandomState
from sub_pixel_response.utils.randomutils import get_randpts


def test_cap():
    """Simple test of points in a spherical cap."""

    deg = np.pi / 180.0

    rs = RandomState(22)
    ra, dec = get_randpts(20.0, 87.0, 2.5, 10000, rng=rs)
    assert np.all(dec > 84.49999)
    assert np.all(dec < 89.50001)
    assert np.all(ra >= 0)
    assert np.all(ra < 360)
    assert len(ra) == 10000
    assert len(dec) == 10000
    x = np.cos(dec * deg) * np.cos(ra * deg)
    y = np.cos(dec * deg) * np.sin(ra * deg)
    z = np.sin(dec * deg)
    x0 = np.cos(87.0 * deg) * np.cos(20.0 * deg)
    y0 = np.cos(87.0 * deg) * np.sin(20.0 * deg)
    z0 = np.sin(87.0 * deg)
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    assert np.all(rho < 0.04363323129985824)
    medrho = np.median(rho)
    assert 0.029 < medrho < 0.032
