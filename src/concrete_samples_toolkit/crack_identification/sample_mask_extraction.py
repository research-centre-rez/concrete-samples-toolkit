from skimage import measure
from tqdm import tqdm
from scipy.optimize import minimize
from skimage.morphology import label

from ..image_fusion.factory_fuser import (
    FuseMethod,
    ImageFuserFactory,
    PercentileFuser,
)

from ..utils.visualisers import imshow

import numpy as np
import cv2 as cv

# Percentile fuser is global so that it does not have to be constantly re-initialised
percentile_fuser: PercentileFuser = ImageFuserFactory.get_fuser(FuseMethod.PERCENTILE)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Function for returning the largest component of a binary mask.

    Args:
        mask (np.ndarray): A binary mask that contains multiple regions
    Returns:
        A binary mask that contains only the largest region of the input mask
    """
    labels_mask = label(1 - mask)

    # Mask contains one or no regions
    if labels_mask.max() <= 1:
        return mask

    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)

    reduced_mask = np.zeros_like(labels_mask)
    reduced_mask[labels_mask == regions[0].label] = 1

    return reduced_mask.astype(np.uint8)


def extract_masks(
    stack: np.ndarray, threshold: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts two types of mask from the input stack, the mask with the largest component that is morphologically closed and the raw mask that you get by threshold filtering.

    Args:
        stack (np.ndarray): Input stack that we want to extract the masks from
        threshold (int): Filtering threshold, the raw mask is extracted as np.min(stack, axis=0) < threshold
    Returns:
        closed_filtered: A mask with the largest region that has been morphologically closed
        raw_mask: raw mask that we got from thresholding the min image
    """
    morph_size = 1  # Init morph size

    raw_mask = np.min(stack, axis=0) < threshold
    closed = np.copy(raw_mask)
    closed_filtered = keep_largest_component(closed)

    # Sometimes there might be "pockets" of non-mask inside the mask, we remove these with adaptive
    # closing to get a single mask that covers the whole sample.
    with tqdm(desc="Adaptive closing", unit=" iterations") as pbar:
        # We check that there is only one background by negating the mask, any 1s inside the sample
        # will be morphologically closed in this loop until there is only one background.
        while np.unique(label(1 - closed_filtered)).size > 2:
            imshow("negation", closed_filtered=(1 - closed_filtered))
            morph_size = morph_size + 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (morph_size, morph_size))

            # Perform closure
            closed_filtered = cv.morphologyEx(closed_filtered, cv.MORPH_CLOSE, kernel)
            pbar.update(1)

    return closed_filtered, raw_mask


def fit_circle(mask: np.ndarray, min_radius: int = 500):
    """
    Function that fits a circle on a binary mask with the use of scipy's `minimize`. The condition that has to be met is that mask == 1 pixels outside of the circle are minimised while mask == 0 pixels inside of the circle are minimised.

    Args:
        mask (np.ndarray): The binary mask that we want to fit a circle to
        min_radius (int): Minimum radius of the circle, measured in pixels
    Returns:
        optimised result that defines the circle.
    """

    def circle_error(params):
        center_x, center_y, radius = params
        y, x = np.where(mask == 1)
        pos = np.sum((x - center_x) ** 2 + (y - center_y) ** 2 > radius**2)
        y, x = np.where(mask == 0)
        neg = np.sum((x - center_x) ** 2 + (y - center_y) ** 2 <= (radius - 1) ** 2)
        return pos + neg

    with tqdm(desc="Fitting circle", unit=" iterations") as pbar:

        def callback(params):
            pbar.update(1)

        opt = minimize(
            circle_error,
            x0=[mask.shape[1] / 2, mask.shape[0] / 2, min_radius * 1.125],
            bounds=(
                (min_radius, mask.shape[1] - min_radius),
                (min_radius, mask.shape[0] - min_radius),
                (min_radius, min_radius * 1.25),
            ),
            method="COBYLA",
            callback=callback,
        )
    return opt


def get_fused_image_pairs(
    video_stack: np.ndarray,
    max_percentile_threshold: int = 95,
    min_percentile_threshold: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns percentile min and max images of the input video stack. Percentile fusion is used so that the user can filter out some of the noise that might be present due to misregistration in the video stack.

    Args:
        video_stack (np.ndarray): input video stack that you wish to extract percentile images from
        max_percentile_threshold (int): max percentile that you want to threshold by
        min_percentile_threshold (int): min percentile that you want to threshold by
    Returns:
        tuple of max and min fused images.
    """

    global percentile_fuser

    max_image = percentile_fuser.get_fused_image(
        video_stack, max_percentile_threshold
    ).astype(np.uint8)

    min_image = percentile_fuser.get_fused_image(
        video_stack, min_percentile_threshold
    ).astype(np.uint8)

    return max_image, min_image


def get_circle_mask(
    video_stack: np.ndarray, threshold: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function for getting a circular mask from the video stack.

    Args:
        video_stack (np.ndarray): Registered stack that you want to extract the mask from.
        threshold (int): Passed into `extract_masks`. Filtering threshold, the raw mask is extracted as np.min(stack, axis=0) < threshold
    Returns:
        A triple of:
            circle_mask (np.ndarray): circular mask of the sample
            mask (np.ndarray): Largest component of the sample mask
            raw_mask (np.ndarray): Raw mask that you get by threshold filtering
    """
    mask, raw_mask = extract_masks(video_stack, threshold)

    circle = fit_circle(mask)
    cx, cy, r = circle.x
    (
        yy,
        xx,
    ) = np.ogrid[: mask.shape[0], : mask.shape[1]]

    circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r**2).astype(np.uint8)

    return circle_mask, mask, raw_mask
