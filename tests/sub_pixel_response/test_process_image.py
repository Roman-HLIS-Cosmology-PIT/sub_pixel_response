import numpy as np

import sub_pixel_response
from sub_pixel_response.process_image import compute_exp_val, compute_pixel_weights


def test_compute_exp_val():
    """
    Test the compute_exp_val function with a simple case where the expected output is known.
    """
    oversam_pix_grid = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    xPower = 2
    yPower = 2
    expected_output = 4
    output = compute_exp_val(oversam_pix_grid, xPower, yPower)
    assert np.allclose(output, expected_output), f"Expected {expected_output}, got {output}"


def test_compute_pixel_weights():
    """
    Tests the compute_pixel_weights function.
    """
    offsets = np.zeros((4096, 4096, 6))
    offsets[0, 0, :] = np.array([1, 0, 0, 0, 1 / 12 * (1 - 1 / 36), 1 / 12 * (1 - 1 / 36)])
    blah = compute_pixel_weights(offsets)
    assert np.allclose(blah[0, 0, :, :], np.ones((6, 6))), f"Expected ones matrix, got {blah[0, 0, :, :]}"
