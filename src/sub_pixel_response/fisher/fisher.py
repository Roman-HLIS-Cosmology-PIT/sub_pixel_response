"""Functions to get the star counts and Fisher information."""

import re

import numpy as np

# constants
wfi_pix = 0.11 / 60.0  # in arcmin
d_telescope = 2.37  # meters

# conversions
ab_minus_vega = {"R": 0.10, "J": 0.91, "H": 1.39, "K": 1.89}
wls = {"R": 0.61, "J": 1.24, "H": 1.60, "K": 2.16}

# Roman information
target_bands = [62, 87, 106, 129, 158, 184, 213]  # in units of 10 nm
areas = [0.779, 0.565, 0.613, 0.615, 0.617, 0.401, 0.387]  # integral A d lambda/lambda in meter^2
peakflux = [0.495, 0.484, 0.440, 0.369, 0.291, 0.214, 0.171]  # max fraction of flux in one pixel
bkgnd = [0.253, 0.254, 0.280, 0.270, 0.292, 0.296, 4.498]  # e/p/s
read = 18.0  # CDS read noise in e


class StarCat:
    """
    Star catalog base class.

    Parameters
    ----------
    infile : str or str-like
        The input file.
    area : float, optional
        The input area in *square arcminutes*. This defaults to the area of Jay Anderson's catalog.
    format : str
        File format type.
    seed : int, optional
        The random number generator seed.

    Methods
    -------
    get_count_rate
        Gets an array of the count rate in e/s for each star in the catalog.
    get_cr_function
        Gets histogram of count rates in the given bins.
    fisher_properties
        Computes Fisher information properties.

    """

    @classmethod
    def _read_header(cls, infile, format):
        """Reads the header from a comment file and gets column values."""

        # get leading commented lines
        lines = []
        with open(infile, "r") as f:
            for line in f:
                if line[0] != "#":
                    break
                lines.append(line.strip())

        # Anderson format
        if format == "Anderson":
            cols = {}
            bands = ["R", "J", "H", "K"]
            mags = ["m606", "jmag_u", "m160_u", "kmag_u"]
            for b in range(len(bands)):
                for line in lines:
                    m = re.search(r" COL(\d+).*" + mags[b] + ":", line)
                    if m:
                        cols[bands[b]] = int(m.group(1)) - 1
            return cols

        return None  # didn't get a header

    def __init__(self, infile, area=31.0, format="Anderson", seed=None):
        self.area = area  # in square arcmin
        self.format = format

        # get the columns
        cols = StarCat._read_header(infile, self.format)

        # now extract the data
        data = np.loadtxt(infile).astype(np.float32)
        self.nstar = np.shape(data)[0]
        self.mags = np.zeros((self.nstar, len(cols)), dtype=np.float32)
        self.bands = k = list(cols.keys())
        for b in range(len(k)):
            self.mags[:, b] = data[:, cols[k[b]]] + ab_minus_vega[k[b]]

        # set up the random number generator
        self.rng = np.random.default_rng(seed=seed)

    def get_count_rate(self, band):
        """
        Gets an array of the count rate in e/s for each star in the catalog.

        Parameters
        ----------
        band : int
            The Roman band (central wavelength in units of 10 nm --- see above table).

        Returns
        -------
        np.ndarray of float
            The count rates in e/s.

        """

        # figure out which index
        ind = -1
        for j in range(len(target_bands)):
            if band == target_bands[j]:
                ind = j
        if ind == -1:
            raise ValueError(f"{ind:d} is not a valid band, choose from {target_bands}.")

        # now interpolate to the band
        i1 = 0
        while i1 < len(self.bands) - 2 and wls[self.bands[i1 + 1]] < 0.01 * band:
            i1 += 1
        frac = np.log(0.01 * band / wls[self.bands[i1]]) / np.log(
            wls[self.bands[i1 + 1]] / wls[self.bands[i1]]
        )
        mag_ab = (1 - frac) * self.mags[:, i1] + frac * self.mags[:, i1 + 1]

        # convert mag_ab to e/s
        return 5.48e10 * areas[ind] * 10 ** (-0.4 * mag_ab)

    def get_cr_function(self, band, bins):
        """
        Gets histogram of count rates in the given bins.

        Parameters
        ----------
        band : int
            The Roman band (central wavelength in units of 10 nm --- see above table).
        bins : np.ndarray of float
            The bin edges in e/s.

        Returns
        -------
        scaledhist : np.ndarray of float
            The histogram, normalized to stars in that bin *per WFI pixel*.
        countrate : np.ndarray of flat
            The geometric mean count rate of the bin (not of the stars in the bin).

        """

        hist, _ = np.histogram(self.get_count_rate(band), bins=bins)
        scaledhist = hist / self.area * wfi_pix**2
        countrate = np.sqrt(bins[1:] * bins[:-1])
        return scaledhist, countrate

    @classmethod
    def _get_peakflux(cls, band):
        """Gets the peak flux in a band."""
        # figure out which index
        ind = -1
        for j in range(len(target_bands)):
            if band == target_bands[j]:
                ind = j
        if ind == -1:
            raise ValueError(f"{ind:d} is not a valid band, choose from {target_bands}.")
        return peakflux[ind]

    @classmethod
    def _get_bkgnd(cls, band):
        """Gets the background in a band."""
        # figure out which index
        ind = -1
        for j in range(len(target_bands)):
            if band == target_bands[j]:
                ind = j
        if ind == -1:
            raise ValueError(f"{ind:d} is not a valid band, choose from {target_bands}.")
        return bkgnd[ind]

    def fisher_properties(self, band, crmin, crmax, t, n_exp):
        """
        Computes Fisher information properties.

        Parameters
        ----------
        band : int
            The Roman band (central wavelength in units of 10 nm --- see above table).
        crmin, crmax : float
            The minimum and maximum count rates for stars to use in e/s.
        t : float
            The equivalent exposure time in s.
        n_exp : int
            Number of exposures.

        Returns
        -------
        ivar_mean, ivar_std : float
            Inverse-variance of the ePSF, mean and standard deviation over a 1 pixel region.

        """

        N = 100  # for integration purposes
        bins = crmin * (crmax / crmin) ** np.linspace(0, 1, N + 1)
        hist, cr = self.get_cr_function(band, bins)
        counts = cr * t
        hist = hist * n_exp

        pkfx = StarCat._get_peakflux(band)
        bg = StarCat._get_bkgnd(band) * t

        # now we want to know the error to which the Delta ePSF can be measured.
        ivar_epsf1 = counts**2 / (pkfx * counts + read**2 + bg)
        ivar_mean = np.sum(hist * ivar_epsf1)
        ivar_std = np.sum(hist * ivar_epsf1**2) ** 0.5

        return ivar_mean, ivar_std

    def fisher_realizations(self, band, crmin, crmax, t, n_exp, fracpix, n_realization):
        """
        Computes realizations of Fisher matrix.

        Parameters
        ----------
        band : int
            The Roman band (central wavelength in units of 10 nm --- see above table).
        crmin, crmax : float
            The minimum and maximum count rates for stars to use in e/s.
        t : float
            The equivalent exposure time in s.
        n_exp : int
            Number of exposures.
        fracpix : float
            Fraction of a pixel to cover in the simulation (e.g., 0.25 if you want to know
            how well the ePSF is measured in a region that is 0.5x0.5 pixels).
        n_realization : int
            Number of realizations.

        Returns
        -------
        ivars : np.ndarray
            Inverse-variance of the ePSF, array of realizations (length `n_realization`).
        sigma_epsf : float
            RMS of the ePSF in the region given by `fracpix`. Note this is not relative
            (uses an absolute ePSF).

        """

        N = 100  # for integration purposes
        bins = crmin * (crmax / crmin) ** np.linspace(0, 1, N + 1)
        hist, cr = self.get_cr_function(band, bins)
        counts = cr * t
        hist = hist * n_exp * fracpix

        pkfx = StarCat._get_peakflux(band)
        bg = StarCat._get_bkgnd(band) * t

        # now we want to know the error to which the Delta ePSF can be measured.
        ivar_epsf1 = counts**2 / (pkfx * counts + read**2 + bg)

        # build the table of realizations
        ivars = np.zeros(n_realization)
        for j in range(n_realization):
            realized_histogram = self.rng.poisson(hist)
            ivars[j] = np.sum(realized_histogram * ivar_epsf1)

        # get the sigma: sqrt <1/ivar>
        sigma_epsf = np.sqrt(np.mean(1.0 / ivars))

        return ivars, sigma_epsf


def fisher(infile, seed=None):
    """
    Builds the Fisher information from a text file.

    Parameters
    ----------
    infile : str or str-like
        The input file.
    seed : int, optional
        The random number generator seed.

    Returns
    -------
    None

    """

    cat = StarCat(infile, seed=seed)
    print("objects per pixel =", cat.nstar / cat.area * wfi_pix**2)
    print(np.round(cat.mags[:8, :], 3))

    N = 20
    bins = np.logspace(1, 3, N + 1)
    hist, cr = cat.get_cr_function(158, bins)
    for j in range(N):
        print(f"{cr[j]:10.4e} {hist[j]:10.4e}")

    for b in [106, 129, 158, 184, 213]:
        print(f"=== {b:03d} ===")

        # now choose a particular size region
        Q = 1e-8 * b / d_telescope / (wfi_pix / 60.0 / 180.0 * np.pi)
        print("Q =", Q)
        fracpix = Q**2 / np.pi

        ivar_mean, ivar_std = cat.fisher_properties(b, 5.0, 1000.0, 63.25, 2900)
        print(ivar_mean * fracpix, ivar_std * np.sqrt(fracpix))
        ivars, sigma_epsf = cat.fisher_realizations(b, 5.0, 1000.0, 63.25, 2900, fracpix, 50000)
        print(np.mean(ivars), np.std(ivars), sigma_epsf)


if __name__ == "__main__":
    fisher("COMPREHENSIVE_2017p38.XYVHJKLM_OBS_EST_W_RDF", seed=42)
