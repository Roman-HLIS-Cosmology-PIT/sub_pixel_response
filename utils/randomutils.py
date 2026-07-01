import numpy as np

def rotation_matrix(ra0,dec0):

    Rz = np.array([
        [np.cos(ra0),-np.sin(ra0),0],
        [np.sin(ra0), np.cos(ra0),0],
        [0,0,1]
    ])

    Ry = np.array([
        [np.sin(dec0),0,np.cos(dec0)],
        [0,1,0],
        [-np.cos(dec0),0,np.sin(dec0)]
    ])

    return Rz @ Ry

def get_ra(y, x):
    return np.pi + np.arctan2(-y, -x)

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
    if rng is None:
        rng = np.random.default_rng()

    phi = rng.uniform(0, 2 * np.pi, npts)
    theta = 2 * np.arcsin(np.sin(np.radians(radius) / 2) * np.sqrt(rng.uniform(0, 1, npts)))
    R = rotation_matrix(np.radians(ra_ctr), np.radians(dec_ctr))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors = np.column_stack((x, y, z))
    rotated_vectors = vectors @ R.T
    x_rot = rotated_vectors[:,0]
    y_rot = rotated_vectors[:,1]
    z_rot = rotated_vectors[:,2]
    ra = get_ra(y_rot, x_rot)
    dec = np.arctan2(z_rot, np.hypot(x_rot, y_rot))
    return ra, dec

