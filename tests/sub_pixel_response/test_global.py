import pytest
from sub_pixel_response import imagesim
from sub_pixel_response.imagesim import GlobalContext


def test_context():
    """Test for the context manager."""

    assert imagesim.GLB_DATA["nside"] == 4088

    with GlobalContext({"nside": 2040}):
        assert imagesim.GLB_DATA["nside"] == 2040

    assert imagesim.GLB_DATA["nside"] == 4088

    with pytest.raises(ValueError):
        with GlobalContext({"6 7": (6, 7)}):
            pass

    assert imagesim.GLB_DATA["nside"] == 4088
