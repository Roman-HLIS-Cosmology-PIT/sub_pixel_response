import numpy as np


def get_randpts(ra_ctr, dec_ctr, radius, npts, rng=None):
    """Get random points within a circle of given radius around a center point (ra_ctr, dec_ctr)"""
    ra = np.random.uniform(ra_ctr - radius, ra_ctr + radius, npts)
    dec = np.random.uniform(dec_ctr - radius, dec_ctr + radius, npts)
    return ra, dec
