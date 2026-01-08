import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk


def extract_binary_crack_mask(
    max_image: np.ndarray,
    min_image: np.ndarray,
    mask: np.ndarray,
    disk_size: int,
    threshold: int,
) -> np.ndarray:
    """
    Overlays the median filtered max and min images for crack extraction. The overlay is done with `np.logical_or`

    Args:
        max_image (np.ndarray): Max image created by image fusion
        min_image (np.ndarray): Min image created by image fusion
        mask (np.ndarray): Binary mask that identifies cracks
        disk_size (int): Determines the size of the disk that will be used for neighbourhood calculation in the median filtering
        threshold (int): Filtering threshold
    Returns:
        A thresholded image (np.ndarray)
    """

    # Create a median filter for noise removal
    max_med = median(max_image, footprint=disk(disk_size), mask=mask)
    min_med = median(min_image, footprint=disk(disk_size), mask=mask)

    minmax_diff_norm = (max_image.astype(float) - max_med.astype(float)) - (
        min_image.astype(float) - min_med.astype(float)
    )

    thresholded_image = np.logical_or(
        minmax_diff_norm * mask > threshold,
        minmax_diff_norm * mask < -threshold,
    )
    return thresholded_image
