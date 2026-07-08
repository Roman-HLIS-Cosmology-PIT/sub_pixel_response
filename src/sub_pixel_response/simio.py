"""Functions for reading data for the simulation."""

import argparse

import pandas as pd
import yaml
from astropy.io import fits


def read_config(config_file):
    """
    Read configuration from YAML file.

    Parameters
    ----------
    config_file : str
        The configuration file location.

    Returns
    -------
    dict
        The configuration file as a Python dictionary.

    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def read_catalog(file_path, file_type=None):
    """
    Read a star catalog and return RA, Dec, and H-band magnitude.

    Parameters
    ----------
    file_path : str
        The input star catalog file.
    file_type : str, optional
        Treat the input file as this type; if not specified, tries to infer the
        type from the file extension.

    Returns
    -------
    dict
        The catalog with keys ``ra``, ``dec``, and ``mag_H``, each representing a
        numpy array. Magnitudes are returned as AB.

    Notes
    -----
    Right now, the function is compatible with two types: the Besançon FITS model
    and the Anderson ASCII model. Anderson models are assumed to be in Vega magnitudes
    and will be converted to AB during reading. Besançon models are in AB as input.

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
    """
    Create argument parser.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        A class containing information on the expected arguments.

    """

    parser = argparse.ArgumentParser(description="Star Simulation Configuration")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    return parser
