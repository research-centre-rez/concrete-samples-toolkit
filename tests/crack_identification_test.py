import numpy as np
import pytest

import concrete_samples_toolkit.crack_identification as mut
from concrete_samples_toolkit.crack_identification import __main__

import concrete_samples_toolkit.crack_identification.sample_mask_extraction as sme
import concrete_samples_toolkit.crack_identification.crack_mask_extraction as cme


def test_extract_binary_crack_mask_basic():
    h, w = 32, 32
    max_image = np.full((h, w), 200, dtype=np.uint8)
    min_image = np.full((h, w), 100, dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)

    out = cme.extract_binary_crack_mask(
        max_image=max_image,
        min_image=min_image,
        mask=mask,
        disk_size=3,
        threshold=10,
    )

    assert out.shape == (h, w)
    assert out.dtype == bool


def test_extract_binary_crack_mask_zero_mask():
    img = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
    mask = np.zeros_like(img)

    out = cme.extract_binary_crack_mask(img, img, mask, disk_size=3, threshold=5)

    assert not out.any()


def test_keep_largest_component():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:9, 5:9] = 1

    # The format that we have is that everything outside of the sample is set to 1, thus we have to invert the mask
    mask = 1 - mask

    out = sme.keep_largest_component(mask)

    assert out.sum() == 16
    assert out.dtype == np.uint8


def test_keep_largest_component_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    out = sme.keep_largest_component(mask)

    assert np.array_equal(out, mask)


def test_get_fused_image_pairs(monkeypatch):
    """
    We are mocking the PercentileFuser
    """

    class DummyFuser:
        def get_fused_image(self, stack, percentile):
            return np.mean(stack, axis=0)

    monkeypatch.setattr(sme, "percentile_fuser", DummyFuser())

    stack = np.random.randint(0, 255, (5, 20, 20), dtype=np.uint8)
    max_img, min_img = sme.get_fused_image_pairs(stack)

    assert max_img.shape == (20, 20)
    assert min_img.shape == (20, 20)
    assert max_img.dtype == np.uint8


def test_overlay_mask_output_shape():
    img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    mask = img > 128

    out = mut.overlay_mask(
        image=img,
        mask=mask,
        alpha=0.5,
        color=(0, 255, 0),
    )

    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8


def test_overlay_mask_invalid_alpha():
    img = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros_like(img)

    with pytest.raises(AssertionError):
        mut.overlay_mask(img, mask, alpha=1.5, color=(255, 0, 0))


def test_convert_binary_img_to_bgr():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:6, 3:6] = 1

    out = mut.convert_binary_img_to_bgr(mask)

    assert out.shape == (10, 10, 3)
    assert np.all(out[3:6, 3:6] == 255)


def test_create_out_filenames(tmp_path):
    fake_npy = tmp_path / "registered.npy"
    fake_npy.touch()

    mask_path, overlay_path = __main__.create_out_filenames(str(fake_npy))

    assert mask_path.endswith(".png")
    assert overlay_path.endswith(".png")
    assert "images" in mask_path
