import argparse

import pandas as pd
import yaml
from astropy.io import fits


def read_config(config_file):
    """Read configuration from YAML file"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def read_catalog(file_path, file_type=None):
    """
    Read a star catalog and return RA, Dec, and H-band magnitude.
    """

    # Determining file type
    if file_type is None:
        file_type = "fits" if file_path.lower().endswith(".fits") else "ascii"

    # FITS catalogs (Besancon)
    if file_type == "fits":
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            ra = data["RAJ2000"]
            dec = data["DECJ2000"]
            mag_H = data["H"]

    # ASCII catalogs (Anderson)
    elif file_type == "ascii":
        with open(file_path, "r") as f:
            for line in f:
                if "RA" in line and "Dec" in line:
                    # Remove the '#' and split the line into a list of names
                    cols = line.replace("#", "").split()
                    break

        df = pd.read_table(file_path, sep=r"\s+", comment="#", names=cols)

        ra = df["RA"]
        dec = df["Dec"]

        # Using Anderson H-band column
        mag_H = df["m160_u"] + 1.39  # This is converting from Vega to AB

    else:
        raise ValueError("Unsupported file type")

    return {"ra": ra, "dec": dec, "mag_H": mag_H}


def make_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description="Star Simulation Configuration")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    return parser
