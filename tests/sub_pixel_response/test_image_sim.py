import numpy as np
from sub_pixel_response.imageSim import compute_poly, l_poly_array, smooth_and_pad, transform_pos


def test_transform_pos():
    """
    Test the transform_pos function with a simple known input.
    """
    x = 1
    y = 2
    expected_output = (3.5, 9.5)
    output = transform_pos(x, y)
    assert np.allclose(output, expected_output), f"Expected {expected_output}, got {output}"


def test_smooth_and_pad():
    """
    Test that smooth_and_pad returns a larger array from added padding.
    """
    input_array = np.ones((4, 4))

    output = smooth_and_pad(input_array, tophatwidth=0)

    assert output.shape[0] > input_array.shape[0]
    assert output.shape[1] > input_array.shape[1]
    assert np.all(np.isfinite(output))


def test_l_poly_array():
    """
    Test l_poly_array for a polynomial of order 0 with known results.
    """
    output = l_poly_array(0, 0.5, 0.5)

    expected = np.array([1.0])
    assert np.allclose(output, expected)


def test_compute_poly():
    """
    Test compute_poly returns a finite, 2D, padded PSF array.
    """
    inpsf_cube = np.ones((1, 4, 4))
    pix = (2044, 2044)
    output = compute_poly(inpsf_cube, pix, order=0)

    assert output.ndim == 2
    assert np.all(np.isfinite(output))
    assert output.shape[0] > 4
    assert output.shape[1] > 4
