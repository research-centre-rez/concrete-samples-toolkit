import argparse
import logging
import os
import sys
import csv
import re
from collections import defaultdict
from datetime import datetime

from . import (
    NGLV,
    BrennerMethod,
)
from tqdm import tqdm
from .metrics import normType
from ..utils import pprint
from ..utils.filename_builder import append_file_extension, create_out_filename


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fused images using two sharpness metrics:\n"
            "- NGLV (Normalized Grey-Level Variance): std_dev^2 / mean, reflects local contrast.\n"
            "- Brenner: sum of squared intensity differences over 2-pixel steps, normalized by image size.\n"
            "Generates separate CSVs per fusion method (MAX, MIN, VAR, MEAN).\n"
            "NOTE: Assumes fused images follow the naming conventions of fuse_video_stack.py.\n"
            "NOTE: Assumes that the registered video stacks follow this naming convention: {sample name}_processing_params.npy"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Paths to fused image(s). Example: -i ../data/images/*.png",
    )
    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want to save the output CSV files (fusion type is auto-appended). If you pass in 'auto' then the program will automatically create an output filename. The file created with auto will have the following name structure: dd-mm_h-m-s_evaluation_{type_of_eval} and will be stored in './evaluation_runs'"
            "Example: --o auto -> ./evaluation_runs/12-07_09-58_evaluation_MAX.csv"
        ),
    )
    required.add_argument(
        "-n",
        "--normalisation_type",
        type=str,
        required=True,
        choices=["l1_norm", "grad_mag"],
        help=(
            "What normalisation do you want to apply to the fused images? The options are 'l1_norm' which applies L1 normalisation and 'grad_mag' which applies gradient magnitude normalisation."
        ),
    )
    return parser.parse_args()


def write_scores_to_csv(csv_filename, columns: list[str], data: list):
    with open(csv_filename, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(columns)
        csv_writer.writerows(data)


def extract_sample_and_fuse_type(
    filename: str, valid_fuse_types: list[str], strip_filename: bool = True
):
    """
    Extracts the sample's name and the fuse type out of the fused image. We assume that the fused image was generated with our `fuse_video_stack.py` and therefore we can remove a lot of useless information for the csv table. If you are passing a filename that was generated differently, use the `strip_filename=False` flag to skip this step.
    Arguments:
        `filename` (str): Name of the fused image you want to extract the sample name and fuse_type from
        `valid_fuse_types` (list[str]): List of all valid fuse types
        `strip_filename` (bool): Automatic removal of useless information from the filename, if you did not use `fuse_video_stack.py` to generate the fused image automatically then set this to False
    Returns:
        (sample_name, fuse_type), if not able to identify the fuse_type it is set as 'UNKNOWN'
    """
    name = os.path.basename(filename)
    base, _ = os.path.splitext(name)
    base_split = base.split("_")

    if strip_filename:
        sample_name = re.sub(r"[-_]part\d+$", " ", base_split[0])
        fuse_type = base_split[-1].upper()
        if fuse_type not in valid_fuse_types:
            fuse_type = "UNKNOWN"

    else:
        sample_name = base_split
        fuse_type = "UNKNOWN"

    return sample_name, fuse_type


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    pprint.log_argparse(args)

    metric_functions = {"NGLV": NGLV(), "Brenner": BrennerMethod()}

    results = defaultdict(list)
    valid_fuse_types = ["MAX", "MIN", "VAR", "MEAN", "unknown"]
    normalisation_type = normType[args.normalisation_type]

    # Store the file as date-month_hour-minute-second_evaluation
    if args.output.lower() == "auto":

        timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
        args.output = f"./evaluation_runs/{timestamp}_evaluation"

    # Create all of the required directories for the output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for filename in tqdm(args.input, desc="Evaluating image"):
        sample_name, fuse_type = extract_sample_and_fuse_type(filename, valid_fuse_types)

        if fuse_type not in valid_fuse_types:
            logger.warning("Skipping unknown fusion type: %s", fuse_type)

        nglv_score = round(
            metric_functions["NGLV"].calculate_metric(
                filename, normalise=True, normalisationType=normalisation_type
            ),
            4,
        )
        brenner_score = round(
            metric_functions["Brenner"].calculate_metric(
                filename, normalise=True, normalisationType=normalisation_type
            ),
            4,
        )

        results[fuse_type].append([sample_name, nglv_score, brenner_score])

    columns = ["Sample name", "NGLV", "Brenner"]
    base_output, _ = os.path.splitext(args.output)
    for fuse_type in sorted(valid_fuse_types):
        score_data = results[fuse_type]
        if not score_data:
            continue

        out_filename = create_out_filename(base_output, [], [fuse_type])
        out_filename = append_file_extension(out_filename, "csv")

        logger.info("Storing evaluation for %s images into %s", fuse_type, out_filename)
        write_scores_to_csv(out_filename, columns, score_data)


if __name__ == "__main__":
    main()
