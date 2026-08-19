import numpy as np
from sub_pixel_response.process_image import (
    compute_exp_val,
    compute_pixel_weights,
    generate_offset_array,
    process_image,
)


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


def test_simple():
    """Simple test with a sine wave."""

    # Set some offsets up, let's try an offset of 1 in the x direction.
    offsets = generate_offset_array([1, 1, 0, 0, 1 / 12 * (1 - 1 / 36), 1 / 12 * (1 - 1 / 36)])

    # Make a simple image with a single Fourier mode.
    im = np.zeros((24576, 24576), dtype=np.float32)
    _s = np.linspace(-2.5, 24572.5, 24576) / 6.0
    for y in range(24576):
        im[y, :] = np.cos(0.25 * _s[y] + 0.1 * _s)

    offset_image = process_image(im, offsets)
    _s2 = np.linspace(0, 4095, 4096)
    im_test = 36 * np.cos(0.25 * _s2[:, None] + 0.1 * (_s2[None, :] + 1))

    print(np.amax(np.abs(offset_image)))
    print(np.amax(np.abs(im_test)))
    print(np.amax(np.abs(offset_image - im_test)))

    assert 35.9 < np.amax(np.abs(offset_image)) < 36.1
    assert np.all(np.abs(offset_image - im_test) < 0.1)
