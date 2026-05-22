import numpy as np
from sub_pixel_response.process_image import compute_exp_val


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
