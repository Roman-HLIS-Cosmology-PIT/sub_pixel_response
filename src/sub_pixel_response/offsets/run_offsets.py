from .. import process_image
from ..imagesim import run_simulation
from ..simio import read_offset_cube


def make_final_image(oversampled_image, offset_file, oversample=6):
    """
    Apply a pixel offset model to an oversampled image.

    Parameters
    ----------
    oversampled_image : np.ndarray
        Oversampled simulator image. The image dimensions are
        imageSize * oversample. imageSize matches the
        detector dimensions represented by the offset map.

    offset_file : str
        FITS file containing the pixel offset cube. The FITS data
        are expected to have shape (6, y, x), with the six
        moment components in the first axis. read_offset_cube
        transposes this to (y, x, 6) for use by process_image.

    oversample : int
        Oversampling factor.

    Returns
    -------
    np.ndarray
        Detector image with the same detector dimensions represented
        by the offset map.
    """
    # FITS order: (6, y, x)
    # In-memory order: (y, x, 6)
    offsets = read_offset_cube(offset_file)

    image_size = offsets.shape[0]

    final_image = process_image.process_image(
        oversampledImage=oversampled_image,
        offsets=offsets,
        imageSize=image_size,
        oversample=oversample,
    )

    return final_image


def run_offset_pipeline(config_path, offset_file):
    """
    Run the simulation and then apply the pixel offset model.
    """

    # Run simulation from imagesim.py
    oversampled_image = run_simulation(str(config_path))

    # Apply offsets and create final image
    final_image = make_final_image(oversampled_image.array, offset_file)

    return final_image

    # if __name__ == "__main__":
    # config_path = "config.yaml"
    # offset_file = "offsets/test_offset_map.fits"


# final_image = run_offset_pipeline(config_path, offset_file)
