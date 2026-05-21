import pandas as pd
from astropy.io import fits


def read_catalog(file_path, file_type=None):
    """
    Read a star catalog and return RA, Dec, and H-band magnitude.
    """

    # Determining file type
    if file_type is None:
        if file_path.lower().endswith(".fits"):
            file_type = "fits"
        else:
            file_type = "ascii"

    # FITS catalogs (Besancon)
    if file_type == "fits":
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            ra = data["RA"]
            dec = data["DEC"]
            mag_H = data["Hmag"]

    # ASCII catalogs (Anderson)
    elif file_type == "ascii":
        try:
            df = pd.read_csv(file_path)
        except:
            df = pd.read_table(file_path, delim_whitespace=True, comment="#")

        ra = df["RA"]
        dec = df["Dec"]

        # Using Anderson H-band column
        mag_H = df["m160_u"]

    else:
        raise ValueError("Unsupported file type")

    return {"ra": ra, "dec": dec, "mag_H": mag_H}
