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
    Extracts a binary crack mask by thresholding a local, median-normalised difference between two fused images (max and min).

    Args:
        max_image (np.ndarray): Max image created by image fusion
        min_image (np.ndarray): Min image created by image fusion
        mask (np.ndarray): Binary mask that identifies cracks
        disk_size (int): Determines the size of the disk that will be used for neighbourhood calculation in the median filtering
        threshold (int): Filtering threshold
    Returns:
        A thresholded image (np.ndarray)

    Note:
        This approach is a bit dubious, but it somehow works. At the moment, there is an alternative method that is being tried. This is located in the branch `alternative_crack_extraction`. Once it is decided that it works better, it will be used instead of this current method.
    """

    # Create a median filter for noise removal
    max_med = median(max_image, footprint=disk(disk_size), mask=mask)
    min_med = median(min_image, footprint=disk(disk_size), mask=mask)

    # For each pixel i,j in the image we compute the local deviation from the median
    #   d_max = max_image - max_med
    #   d_min = min_image - min_med
    # Then we compute their differences:
    #   minmax_diff_norm = d_max - d_min = (max_image - max_med) - (min_image - min_med)
    # This highlights pixels where the bright peaks (relative to local background) differ from
    # dark-throughs (relative to local background), the theory is that the cracks will produce
    # a characteristic sign and magnitude in this difference
    minmax_diff_norm = (max_image.astype(float) - max_med.astype(float)) - (
        min_image.astype(float) - min_med.astype(float)
    )

    # A pixel is declared a crack if the masked difference exceeds a threshold in absolute value
    thresholded_image = np.logical_or(
        minmax_diff_norm * mask > threshold,
        minmax_diff_norm * mask < -threshold,
    )
    return thresholded_image
