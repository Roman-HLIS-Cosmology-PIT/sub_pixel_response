"""Tests for assorted utilities."""

import numpy as np
from sub_pixel_response.multi_sca_utils import euler_angle_conversion_w, r_sca, r_wfi


def test_multi_sca_utils():
    """Some simple tests."""

    for sca in range(1, 19):
        r = r_sca(sca)
        assert np.abs(np.linalg.det(r) - 1) < 1.0e-10  # unit determinant
        s = r @ r.T - np.identity(3)
        assert np.all(np.abs(s) < 1.0e-12)  # check orthogonal
        assert 2.99 < np.linalg.trace(r) < 3.00  # didn't rotate too much


def test_r_wfi():
    """Build some rotation matrices with simple test cases."""

    assert np.allclose(r_wfi(0, np.pi / 2, 3 * np.pi / 2), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    assert np.allclose(r_wfi(0, np.pi / 2, np.pi), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    assert np.allclose(r_wfi(0, 0, np.pi), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert np.allclose(r_wfi(0, -np.pi / 2, np.pi), np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

    # space around the equator
    assert np.allclose(r_wfi(np.pi / 2, 0, np.pi), np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    assert np.allclose(r_wfi(np.pi, 0, np.pi), np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))
    assert np.allclose(r_wfi(3 * np.pi / 2, 0, np.pi), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    # now around the equator, upside-down
    assert np.allclose(r_wfi(0, 0, 0), np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
    assert np.allclose(r_wfi(np.pi / 2, 0, 0), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    assert np.allclose(r_wfi(np.pi, 0, 0), np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    assert np.allclose(r_wfi(3 * np.pi / 2, 0, 0), np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]))

    # now around the equator, lonpole = pi/2 so "x wfi" is North
    assert np.allclose(r_wfi(0, 0, np.pi / 2), np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
    assert np.allclose(r_wfi(np.pi / 2, 0, np.pi / 2), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    assert np.allclose(r_wfi(np.pi, 0, np.pi / 2), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    assert np.allclose(r_wfi(3 * np.pi / 2, 0, np.pi / 2), np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))

    # orthogonality tests for lots of orientations
    for j in range(1000):
        alpha = 0.2 * np.pi * (j // 100 - 0.5)
        delta = 0.1 * np.pi * ((j % 100) // 10 - 4.5)
        phi = 0.2 * np.pi * (j % 10 - 4.5)
        r = r_wfi(alpha, delta, phi)
        assert np.allclose(r @ r.T, np.identity(3))
        assert np.linalg.det(r) > 0

        # does the back-conversion work?
        a, d, p = euler_angle_conversion_w(r)
        assert np.cos(a - alpha) > 0.99999
        assert np.cos(d - delta) > 0.99999
        assert np.cos(p - phi) > 0.99999
