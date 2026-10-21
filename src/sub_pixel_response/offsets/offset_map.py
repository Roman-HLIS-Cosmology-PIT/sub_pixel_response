import os

import fitsio
import numpy as np
from fitsio import FITSHDR


def make_im_offset(offset_pattern=1):
    """Makes toy offsets.

    Parameters
    ----------
    offset_pattern : int, optional
        Which pattern to generate.

    Returns
    -------
    np.ndarray of float
        A toy offset map. Shape (6, 4088, 4088).
    """

    nside = 4088
    n_perturbations = 6

    block_size = 64
    offset_value = 0.02

    # Empty offset cube
    im_offset = np.zeros(
        (n_perturbations, nside, nside),
        dtype=np.float32,
    )

    # Checkerboard pattern (something simple and recognizable to start)
    if offset_pattern == 1:
        for y in range(0, nside, block_size):
            for x in range(0, nside, block_size):
                block_y = y // block_size
                block_x = x // block_size

                if (block_x + block_y) % 2 == 0:
                    im_offset[
                        0,
                        y : y + block_size,
                        x : x + block_size,
                    ] = offset_value

                else:
                    im_offset[
                        0,
                        y : y + block_size,
                        x : x + block_size,
                    ] = -offset_value

    return im_offset


if __name__ == "__main__":
    # Offset cube
    hdr = FITSHDR()
    outfile = "src/sub_pixel_response/offsets/test_offset_map.fits"
    im_offset = make_im_offset()
    fitsio.write(str(outfile), im_offset, header=hdr, clobber=True)

    print(f"Offset map saved to: {os.path.abspath(outfile)}")
    print(f"Shape: {im_offset.shape}")
    print(f"Data type: {im_offset.dtype}")
    print(f"File exists: {os.path.exists(outfile)}")
