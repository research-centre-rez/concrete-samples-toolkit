import numpy as np
import cv2 as cv
from skimage import measure
from skimage.filters.rank import median
from skimage.morphology import disk, label

from ..utils.visualisers import imshow

from concrete_samples_toolkit.crack_identification.sample_mask_extraction import (
    extract_masks,
)
from concrete_samples_toolkit.image_fusion.crop_image import crop_image
from concrete_samples_toolkit.image_fusion.factory_fuser import (
    FuseMethod,
    ImageFuserFactory,
    PercentileFuser,
)

percentile_fuser: PercentileFuser = ImageFuserFactory.get_fuser(FuseMethod.PERCENTILE)


def extract_median_binary_mask(
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


def extract_sobel_binary_mask(
    registered_stack: np.ndarray | str, config: dict[str, int]
) -> np.ndarray:
    global percentile_fuser

    if isinstance(registered_stack, np.ndarray):
        input_stack = registered_stack
    elif isinstance(registered_stack, str):
        input_stack = np.load(registered_stack)

    median_image = percentile_fuser.get_fused_image(input_stack)
    mask, _ = extract_masks(input_stack, config["masking_threshold"])

    median_image[mask == 0] = 0
    median_image = crop_image(median_image, mask)
    median_image = cv.GaussianBlur(
        median_image, (config["gauss_kernel"], config["gauss_kernel"], 0)
    )
    sobel_x = cv.Sobel(median_image, cv.CV_64F, 1, 0, ksize=config["sobel_kernel"])
    sobel_y = cv.Sobel(median_image, cv.CV_64F, 1, 0, ksize=config["sobel_kernel"])

    thresholded_sobel_x = np.zeros_like(sobel_x)
    thresholded_sobel_x[sobel_x < config["crack_threshold"]] = 1

    thresholded_sobel_y = np.zeros_like(sobel_y)
    thresholded_sobel_y[sobel_y < config["crack_threshold"]] = 1

    # Showing the x and y derivatives in one image, change to True if you want to see the image
    if False:
        bgr_image = np.zeros_like(median_image, dtype=np.uint16)
        bgr_image = cv.cvtColor(bgr_image, cv.COLOR_GRAY2BGR)

        bgr_image[:, :, 0] = thresholded_sobel_x * 255
        bgr_image[:, :, 2] = thresholded_sobel_y * 255

        imshow("blue channel = gx, red channel = gy", channeled_image=bgr_image)

    combined_x_and_y_derivatives = np.logical_or(
        thresholded_sobel_x, thresholded_sobel_y
    )

    # Filtering out small regions

    labeled_mask = label(combined_x_and_y_derivatives)
    regions_of_mask = measure.regionprops(labeled_mask)

    filtered_mask = np.zeros_like(combined_x_and_y_derivatives)
    for region in regions_of_mask:
        if region.area > config["area_threshold"]:
            filtered_mask[labeled_mask == region.label] = 1

    return filtered_mask
