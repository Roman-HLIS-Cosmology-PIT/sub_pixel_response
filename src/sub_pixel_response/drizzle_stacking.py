import sys
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from drizzle.resample import Drizzle
from drizzle.utils import calc_pixmap


def drizzilicious(input_dir, out_nx=4000, out_ny=4000):
    """
    Stacks FITS images from an input directory using the Drizzle stacking method.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the input FITS files.
    out_nx : int, optional
        Number of pixels in the x-direction for the output image.
    out_ny : int, optional
        Number of pixels in the y-direction for the output image.

    Returns
    -------
    final_image : np.ndarray
        Final stacked image.
    final_weight : np.ndarray
        Final weight map.
    final_context : np.ndarray
        Final context map.
    """
    # make a for loop to iterate through all the fits files in the input_dir and apply drizzle

    print(f"Drizzle stacking process started for directory: {input_dir}")
    input_dir_path = Path(input_dir)
    input_files = sorted(input_dir_path.glob("final_image*.fits"))
    base_file = input_dir_path / "final_image.fits"

    input_files.remove(base_file)
    input_files.insert(0, base_file)  # Ensure the base file is first in the list

    with fits.open(base_file) as hdul:
        output_wcs = WCS(hdul[1].header)
    driz = Drizzle(out_shape=(out_ny, out_nx), kernel="square", fillval=0.0)
    # Value for output pixels with no coverage
    total_exptime = 0.0

    for file_path in input_files:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            input_wcs = WCS(hdul[1].header)
            exptime = hdul[1].header["EXPTIME"]
            pixmap = calc_pixmap(input_wcs, output_wcs, shape=data.shape)
            driz.add_image(data=data, exptime=exptime, pixmap=pixmap, iscale=1.0, pixel_scale_ratio=1.0)
            # Scale factor if input/output pixel scales differ (or use pixel_scale_ratio)
        total_exptime += exptime

    final_image = driz.out_img
    final_weight = driz.out_wht
    final_context = driz.out_ctx

    output_header = output_wcs.to_header()
    output_header["EXPTIME"] = total_exptime

    output_path = input_dir_path / "drizzled_output.fits"

    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=final_image, header=output_header, name="SCI"),
            fits.ImageHDU(data=final_weight, name="WHT"),
            fits.ImageHDU(data=final_context, name="CTX"),
        ]
    ).writeto(output_path, overwrite=True)

    # Here you would add the actual implementation of the drizzle stacking algorithm
    # For example, loading images, aligning them, and combining them into a final image.
    # need to write this to a FITS file
    print("Drizzle stacking process completed.")
    print("Saving output to 'drizzled_output.fits'...")
    return final_image, final_weight, final_context


if __name__ == "__main__":
    drizzilicious(sys.argv[1])
