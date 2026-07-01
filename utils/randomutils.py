import numpy as np


def get_randpts(ra_ctr, dec_ctr, radius, npts, rng=None):
    """
    Get random points within a circle of given radius around a center point (ra_ctr, dec_ctr).

    Parameters
    ----------
    ra_ctr : float
        Right Ascension of the center point in degrees.
    dec_ctr : float
        Declination of the center point in degrees.
    radius : float
        Radius of the circle in degrees.
    npts : int
        Number of random points to generate.
    rng : np.random.Generator, optional
        A random number generator instance. If None, the default numpy random generator is used.

    Returns
    -------
    ra : np.ndarray
        Array of random Right Ascension values.
    dec : np.ndarray
        Array of random Declination values.
    """
    ra = np.random.uniform(ra_ctr - radius, ra_ctr + radius, npts)
    dec = np.random.uniform(dec_ctr - radius, dec_ctr + radius, npts)
    return ra, dec
