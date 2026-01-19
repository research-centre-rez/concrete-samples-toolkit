import sys
import argparse
import cv2 as cv
import numpy as np
from skimage import measure
from skimage.morphology import label

from concrete_samples_toolkit.crack_identification.__main__ import overlay_mask
from concrete_samples_toolkit.image_fusion import crop_image

from ..image_fusion.factory_fuser import FuseMethod, PercentileFuser, ImageFuserFactory

from ..utils.visualisers import imshow
from .sample_mask_extraction import extract_masks

import matplotlib.pyplot as plt


def parse_args():
    argparser = argparse.ArgumentParser()

    required = argparser.add_argument_group("required arguments")

    required.add_argument("-i", "--input", type=str, required=True)

    return argparser.parse_args()


def main():
    args = parse_args()

    fuser: PercentileFuser = ImageFuserFactory.get_fuser(FuseMethod.PERCENTILE)

    npy_file = np.load(args.input)

    median_image = fuser.get_fused_image(npy_file)
    mask, _ = extract_masks(npy_file, threshold=10)
    # imshow(mask=mask)

    median_image[mask == 0] = 0
    median_image = crop_image(median_image, mask)
    median_image = cv.GaussianBlur(
        median_image, (3, 3), 0
    )  # Smoothing to make edge detection better
    gx = cv.Sobel(median_image, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(median_image, cv.CV_64F, 0, 1, ksize=3)

    threshold = -11
    thresh_gx = np.zeros_like(gx)
    thresh_gx[gx < threshold] = 1

    thresh_gy = np.zeros_like(gy)
    thresh_gy[gy < threshold] = 1

    bgr_image = np.zeros_like(median_image, dtype=np.uint16)
    bgr_image = cv.cvtColor(bgr_image, cv.COLOR_GRAY2BGR)

    bgr_image[:, :, 0] = thresh_gy * 255
    bgr_image[:, :, 2] = thresh_gx * 255

    imshow("blue channel = gx, red channel = gy", channeled_image=bgr_image)

    # imshow(threshold_gx=thresh_gx, threshold_gy=thresh_gy)
    # plt.hist(gx.ravel(), 200, range=[-100,100])
    # plt.hist(gy.ravel(), 200, range=[-100,100])
    # plt.show()

    combined = np.logical_or(thresh_gy, thresh_gx)

    # thresholded_mag = np.zeros_like(magnitude)
    # threshold = 9
    # thresholded_mag[magnitude > threshold] = 1
    imshow(
        threshold_gx=thresh_gx,
        threshold_gy=thresh_gy,
        median_image=median_image,
        magnitude=combined,
    )

    imshow(median_image=median_image, magnitude=combined)


    labels_overlaid = label(combined)
    regions = measure.regionprops(labels_overlaid)

    plt.hist([r.area for r in regions], 200, range=[0, 300])
    plt.show()

    region_threshold = 80
    reduced_mask = np.zeros_like(combined)
    for reg in regions:
        if reg.area > region_threshold:
            reduced_mask[labels_overlaid == reg.label] = 1

    imshow(original_mask=combined, reduced_mask=reduced_mask)

    overlaid = overlay_mask(
        image=median_image, mask=reduced_mask, alpha=0.5, color=[0, 0, 255]
    )
    imshow(overlaid_mask=overlaid)


if __name__ == "__main__":
    main()
