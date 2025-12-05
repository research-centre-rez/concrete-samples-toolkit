import argparse
import os
from typing import Generator, Union
import csv
import sys
import logging
import json

import numpy as np
import cv2 as cv
from skimage import measure
from tqdm import tqdm
from scipy.optimize import minimize
from skimage.filters.rank import median
from skimage.morphology import disk, binary_erosion, closing, label
import matplotlib.pyplot as plt


from concrete_samples_toolkit.image_fusion.factory_fuser import (
    FuseMethod,
    ImageFuserFactory,
    PercentileFuser,
)
from concrete_samples_toolkit.utils.filename_builder import (
    append_file_extension,
    create_out_filename,
)
from concrete_samples_toolkit.utils.visualisers import imshow

logger = logging.getLogger(__name__)
percentile_fuser: PercentileFuser = ImageFuserFactory.get_fuser(FuseMethod.PERCENTILE)


def parse_args():
    argparser = argparse.ArgumentParser()
    required = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the csv that contains sample pairings of before + after scans. You can generate this csv with utils/create_csv_pairs.py",
    )

    optional.add_argument(
        "-o", 
        "--output",
        type=str,
        required=False,
        help="How to save the resulting csv and json files."
    )

    return argparser.parse_args()


def parse_csv_pairs(csv_file_path: str) -> Generator[list[str], None, None]:

    with open(csv_file_path, "r+") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skipping header rows

        yield from csv_reader


def keep_largest_component(mask):
    labels_mask = label(1 - mask)
    if labels_mask.max() == 0:
        return mask

    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for region in regions[1:]:
            labels_mask[region.coords[:, 0], region.coords[:, 1]] = 0

    labels_mask[labels_mask != 0] = 1

    return (1 - labels_mask).astype(np.uint8)


def adaptive_closing(stack, threshold=10):
    # Adaptive closing
    # - increases size of the circular kernel until only one segment remains
    morph_size = 1
    raw_mask = np.min(stack, axis=0) < threshold
    closed = np.copy(raw_mask)
    closed_filtered = keep_largest_component(closed)

    with tqdm(desc="Adaptive closing", unit=" iterations") as pbar:
        while np.unique(label(1 - closed_filtered)).size > 2:
            morph_size = morph_size + 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (morph_size, morph_size))
            # Perform closure
            closed_filtered = cv.morphologyEx(
                (np.min(stack, axis=0) >= 10).astype(np.uint8), cv.MORPH_CLOSE, kernel
            )
            pbar.update(1)

    return 1 - closed_filtered, morph_size, 1 - raw_mask


def fit_circle(mask, min_radius=500):
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


def get_thresholded_image(
    max_image, min_image, mask, disk_size, threshold
) -> np.ndarray:

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


def get_fused_image_pairs(
    video_stack: np.ndarray,
    max_percentile_threshold: int,
    min_percentile_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:

    global percentile_fuser

    max_image = percentile_fuser.get_fused_image(
        video_stack, max_percentile_threshold
    ).astype(np.uint8)

    min_image = percentile_fuser.get_fused_image(
        video_stack, min_percentile_threshold
    ).astype(np.uint8)

    return max_image, min_image


def get_circle_mask(
    video_stack: np.ndarray, threshold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask, k_size, raw_mask = adaptive_closing(video_stack, threshold)

    circle = fit_circle(mask)
    cx, cy, r = circle.x
    (
        yy,
        xx,
    ) = np.ogrid[: mask.shape[0], : mask.shape[1]]

    circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r**2).astype(np.uint8)

    return circle_mask, mask, raw_mask


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float, color: tuple[int]
) -> np.ndarray:
    int_image = image.astype(np.uint8)

    bgr_image = cv.cvtColor(int_image, cv.COLOR_GRAY2BGR)

    mask_bgr = np.zeros_like(bgr_image)
    mask_bgr[mask] = color

    blended = cv.addWeighted(bgr_image, 1.0, mask_bgr, alpha, 0)
    return blended


def create_out_filenames(path):

    root_dir, base_name = os.path.split(path)
    base_name = base_name.replace(".npy", "")
    image_destination = os.path.join(root_dir, "images")
    os.makedirs(image_destination, exist_ok=True)

    mask_out_filename = os.path.join(image_destination, base_name)
    mask_out_filename = create_out_filename(mask_out_filename, ["binary", "mask"], [])
    mask_out_filename = append_file_extension(mask_out_filename, ".png")

    overlaid_mask_filename = os.path.join(image_destination, base_name)
    overlaid_mask_filename = create_out_filename(
        overlaid_mask_filename, ["binary", "mask", "overlaid"], []
    )
    overlaid_mask_filename = append_file_extension(overlaid_mask_filename, ".png")

    return mask_out_filename, overlaid_mask_filename


def convert_binary_img_to_grayscale(image: np.ndarray) -> np.ndarray:

    int_image = image.astype(np.uint8)
    int_image[int_image != 0] = 255

    bgr_image = cv.cvtColor(int_image, cv.COLOR_GRAY2BGR)
    return bgr_image


def process_pair_of_npy_files(
    before_file: str, after_file: str
) -> dict[str, np.ndarray]:

    before_exposure = np.load(before_file)
    after_exposure = np.load(after_file)

    max_threshold = 95
    min_threshold = 5
    mask_threshold = 5

    before_max_image, before_min_image = get_fused_image_pairs(
        before_exposure, max_threshold, min_threshold
    )

    after_max_image, after_min_image = get_fused_image_pairs(
        after_exposure, max_threshold, min_threshold
    )

    _, before_circle_mask, _ = get_circle_mask(before_exposure, mask_threshold)
    _, after_circle_mask, _ = get_circle_mask(after_exposure, mask_threshold)

    before_minmax_diff_norm = get_thresholded_image(
        before_max_image,
        before_min_image,
        before_circle_mask,
        disk_size=51,
        threshold=10,
    )

    after_minmax_diff_norm = get_thresholded_image(
        after_max_image, after_min_image, after_circle_mask, disk_size=51, threshold=10
    )

    before_max_overlaid = overlay_mask(
        before_max_image, before_minmax_diff_norm, 0.5, [0, 255, 0]
    )

    after_max_overlaid = overlay_mask(
        after_max_image, after_minmax_diff_norm, 0.5, [255, 0, 0]
    )

    before_mask_out_filename, before_overlaid_mask_filename = create_out_filenames(
        before_file
    )
    after_mask_out_filename, after_overlaid_mask_filename = create_out_filenames(
        after_file
    )

    cv.imwrite(
        before_mask_out_filename,
        convert_binary_img_to_grayscale(before_minmax_diff_norm),
    )
    cv.imwrite(before_overlaid_mask_filename, before_max_overlaid)

    cv.imwrite(
        after_mask_out_filename, convert_binary_img_to_grayscale(after_minmax_diff_norm)
    )
    cv.imwrite(after_overlaid_mask_filename, after_max_overlaid)

    logger.info(f"Before exp area: {before_minmax_diff_norm.sum()}")
    logger.info(f"After exp area: {after_minmax_diff_norm.sum()}")

    processed_images = {
        "before_minmax_diff_norm": before_mask_out_filename,
        "after_minmax_diff_norm": after_mask_out_filename,
        "before_area": before_minmax_diff_norm.sum(),
        "after_area": after_minmax_diff_norm.sum(),
    }

    return processed_images


def write_metrics_into_json(
    before_path: str,
    after_path: str,
    processed_pair: dict[str, np.ndarray],
    out_path: str,
) -> None:

    sample_name = before_path.split("/")[1:3]
    sample_name = "_".join(sample_name)

    new_entry = {
        "before_exposure_path": before_path,
        "after_exposure_path": after_path,
        "metrics": {
            "before_exposure_area": int(processed_pair["before_area"]),
            "after_exposure_area": int(processed_pair["after_area"]),
        },
        "images": {
            "before_mask_path": processed_pair["before_minmax_diff_norm"],
            "after_mask_path": processed_pair["after_minmax_diff_norm"],
        },
    }

    json_path = append_file_extension(out_path, "json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[sample_name] = new_entry

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    csv_path = append_file_extension(out_path, ".csv")
    csv_exists = os.path.exists(csv_path)

    row = {
        "sample_name": sample_name,
        "before_exposure_path": before_path,
        "after_exposure_path": after_path,
        "before_exposure_area": int(processed_pair["before_area"]),
        "after_exposure_area": int(processed_pair["after_area"]),
        "before_mask_path": processed_pair["before_minmax_diff_norm"],
        "after_mask_path": processed_pair["after_minmax_diff_norm"],
    }

    # Append or create
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f'Entry {sample_name} has been processed.')

    return None


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    out_filename = args.output if args.output is not None else "samples_crack_area.json"
    out_filename = os.path.join(os.path.dirname(args.input), out_filename)

    for before_exp, after_exp in parse_csv_pairs(args.input):
        base_dir = os.path.dirname(args.input)
        before_exp_joined = os.path.join(base_dir, before_exp)
        after_exp_joined = os.path.join(base_dir, after_exp)
        processed_result = process_pair_of_npy_files(
            before_exp_joined, after_exp_joined
        )

        write_metrics_into_json(before_exp, after_exp, processed_result, out_filename)


if __name__ == "__main__":
    main()
