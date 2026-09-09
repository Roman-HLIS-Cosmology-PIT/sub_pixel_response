import sub_pixel_response.process_image as process_image
from sub_pixel_response.imagesim import run_simulation
from sub_pixel_response.simio import read_offset_cube


def make_final_image(oversampled_image, offset_file, oversample=6):
    """
    Apply a pixel offset model to an oversampled image.

    Parameters
    ----------
    oversampled_image : np.ndarray
        Oversampled simulator image.

    offset_file : str
        FITS file containing the pixel offset cube.

    oversample : int
        Oversampling factor.

    Returns
    -------
    np.ndarray
        Final 4088 x 4088 detector image.
    """

    offsets = read_offset_cube(offset_file)

    image_size = offsets.shape[0]

    detector_image = process_image.process_image(
        oversampledImage=oversampled_image,
        offsets=offsets,
        imageSize=image_size,
        oversample=oversample,
    )

    # Remove the 4-pixel reference-pixel border
    final_image = detector_image[4:-4, 4:-4]

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
    config_path = "config.yaml"
    offset_file = "offsets/test_offset_map.fits"

    final_image = run_offset_pipeline(config_path, offset_file)
