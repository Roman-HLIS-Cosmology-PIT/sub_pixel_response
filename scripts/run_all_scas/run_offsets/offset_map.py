import fitsio
import numpy as np
from fitsio import FITSHDR

nside = 4096
n_perturbations = 6

offset_pattern = 1
outfile = "offsets/test_offset_map.fits"

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

# Reference pixels (zero in this first map)
im_offset[:, :, :4] = 0.0
im_offset[:, :, -4:] = 0.0
im_offset[:, :4, :] = 0.0
im_offset[:, -4:, :] = 0.0


# Offset cube
hdr = FITSHDR()
fitsio.write(outfile, im_offset, header=hdr, clobber=True)
