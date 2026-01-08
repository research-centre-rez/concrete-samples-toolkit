import argparse
import os
from typing import Generator
import csv
import logging
import json

import numpy as np
import cv2 as cv

from .sample_mask_extraction import (
    get_fused_image_pairs,
    get_circle_mask,
)
from .crack_mask_extraction import extract_binary_crack_mask


from ..utils.filename_builder import (
    append_file_extension,
    create_out_filename,
)

logger = logging.getLogger(__name__)


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for identifying cracks in the registered concrete samples. For each sample, it will create the following columns: \n sample_name: name of the sample \n before/after_expo: Path to the registered stack for before and after exposition of the sample \n before/after_exposure_area: Area of the identified cracks of the before/after sample \n before/after_mask_path: Path to the binary masks that show the identified cracks, for both before and after samples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    required = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the csv that contains sample pairings of before and after scans. You can generate this csv with utils/create_csv_pairs.py.",
    )

    optional.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="How to save the resulting csv and json files. If you don't use this argument, the name is automatically set as 'samples_crack_area.json'.",
    )

    return argparser.parse_args()


def parse_csv_pairs(csv_file_path: str) -> Generator[list[str], None, None]:
    """
    Function responsible for parsing pairs from the input csv.

    Args:
        csv_file_path (str): Path to the csv, should contain `.csv`
    Yields:
        a tuple, where the first element is the path to the registered sample before exposure and the second element is the path to registered sample after exposure.
    """

    with open(csv_file_path, "r+") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skipping the header row

        yield from csv_reader


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float, color: tuple[int, int, int]
) -> np.ndarray:
    """
    A function that will overlay the exctracted cracks over the (fused) image. Mainly used for debugging purposes to check the validity of the generated binary mask.

    Args:
        image (np.ndarray): Fused image that will be overlaid
        mask (np.ndarray): Mask that we want to show
        alpha (float): intensity of the mask, should be in range [0,1]
        color (tuple[int]): Colour of the mask, would be passed as a triple where each element represents the intensity of the individual channels
    Returns:
        Fused image that has the mask overlaid on top of it
    """

    assert len(color) == 3, "The mask colour should have 3 channels"
    assert 0 < alpha < 1, "Alpha should be in range [0,1]"

    int_image = image.astype(np.uint8)

    bgr_image = cv.cvtColor(int_image, cv.COLOR_GRAY2BGR)

    mask_bgr = np.zeros_like(bgr_image)
    mask_bgr[mask] = color

    blended = cv.addWeighted(bgr_image, 1.0, mask_bgr, alpha, 0)
    return blended


def create_out_filenames(path: str) -> tuple[str, str]:
    """
    Given a path to a registered `.npy` stack, create ouput filenames for the masks. Creates a filename for both the crack binary mask and the crack binary mask overlaid with the image fusion.

    Args:
        path (str): path to the registered `.npy` stack
    Returns:
        tuple of:
            mask_out_filename (str): filename for the mask image
            overlaid_mask_filename (str): filename for the overlaid image
    """

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


def convert_binary_img_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Converts a binary image into a bgr image.

    Args:
        image (np.ndarray): Binary image
    Returns:
        BGR representation of the binary image
    """

    int_image = image.astype(np.uint8)
    int_image[int_image != 0] = 255

    bgr_image = cv.cvtColor(int_image, cv.COLOR_GRAY2BGR)
    return bgr_image


def process_pair_of_npy_files(
    before_file: str, after_file: str
) -> dict[str, np.ndarray]:
    """
    This function processes one sample's before and after exposure video stacks, it saves the masks in the same directory as the npy file is saved. Returns a dictionary that contains the paths and the area of the cracks for the sample.

    Args:
        before_file (str): Path to before exposure `.npy` stack.
        after_file (str): Path to before exposure `.npy` stack.
    Returns:
        dictionary that contains the following entries:
            "before_minmax_diff_norm": mask before the exposure of sample
            "before_minmax_diff_norm": mask after the exposure of the sample
            "before_area": crack area of the before exposure
            "after_area": crack area of the after exposure
    """

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

    before_minmax_diff_norm = extract_binary_crack_mask(
        before_max_image,
        before_min_image,
        before_circle_mask,
        disk_size=51,
        threshold=10,
    )

    after_minmax_diff_norm = extract_binary_crack_mask(
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
        convert_binary_img_to_bgr(before_minmax_diff_norm),
    )
    cv.imwrite(before_overlaid_mask_filename, before_max_overlaid)

    cv.imwrite(
        after_mask_out_filename, convert_binary_img_to_bgr(after_minmax_diff_norm)
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


def write_metrics_into_json_csv(
    before_path: str,
    after_path: str,
    processed_pair: dict[str, np.ndarray],
    out_path: str,
) -> None:
    """
    Writes the information about a sample into a JSON and CSV. The information that is written is the location of the registered `.npy` files, the cracks that were identified, and their area.

    Args:
        before_path (str): Path to the registered `.npy` stack *before* the radiation exposure
        after_path (str): Path to the registered `.npy` stack *after* the radiation exposure
        processed_pair (dict[str, np.ndarray]): Processed pair obtained from `process_pair_of_npy_files()`
        out_path (str): Filename of the JSON/CSV
    Returns:
        None, the two files are written into disk.
    """

    # The samples are stored in the following way: {before|after}_exp/sample_name/face/illumination_angle/registered.npy
    # thus we only care about the information from the sample name to the illumination angle
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

    # I think this can be done differently but I am not sure how
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

    logger.info(f"Entry {sample_name} has been processed.")

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

        write_metrics_into_json_csv(
            before_exp, after_exp, processed_result, out_filename
        )


if __name__ == "__main__":
    main()
