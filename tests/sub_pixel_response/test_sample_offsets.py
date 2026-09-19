import numpy as np
from sub_pixel_response.offsets.offset_map import make_im_offset


def test_make_im_offset():
    """Simple test for the offset map generator."""

    map = make_im_offset(offset_pattern=1)
    assert np.shape(map) == (6, 4088, 4088)
    assert np.all(np.abs(map[0]) > 1e-12)
