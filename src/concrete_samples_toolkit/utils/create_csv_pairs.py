import argparse
from collections import defaultdict
import logging
import os
import csv

from pprint import pprint_dict
import re


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the root of the directory you want to create pairs of.",
    )

    return parser.parse_args()


def find_npy_files(exp_subdir: str, base_dir: str) -> dict[str, list]:
    """
    For each sample that has been scanned in a directory, this function will find all the registered `.npy` files that "belong" under that sample for one type of exposure (so either before or after exposure).

    The tree structure for one exposure time should look something like the following:
        after_exp/
            sample_name/
                front/
                    video_name/
                        registered_video.npy    <---
                        images/                    |
                back/                              |--- We are after these
                    video_name/                    |
                        registered_video2.npy   <---
                        images/
        before_exp/
            similarly...

    Args:
        exp_subdir (str): The exposure subdirectory we want to find .npy files in
        base_dir (str): Root of the dataset, since the resulting csv contains only relative paths from the root of the dataset, everything before it is removed.
    Returns:
        A dictionary that contains for each sample face, the list of all `.npy` files.
    """
    found_npy_files = {}
    for current_dir, subdirectories, _ in os.walk(exp_subdir):
        if "front" in subdirectories:
            sample_name = current_dir.split("/")[-1]
            for face_subdir in subdirectories:
                # Used for indexing
                sample_face = os.path.join(sample_name, face_subdir)
                found_npy_files[sample_face] = []

                sample_subdir = os.path.join(current_dir, face_subdir)

                for current_face_dir, _, face_files in os.walk(sample_subdir):
                    # We only care about the relative path from the sample's subdirectories
                    out_dir = current_face_dir.replace(base_dir, "")
                    npy_files = [
                        os.path.join(out_dir, npy_file)
                        for npy_file in face_files
                        if npy_file.lower().endswith(".npy")
                    ]
                    if npy_files:
                        found_npy_files[sample_face].extend(npy_files)

    return found_npy_files


def match_before_and_after_npy_files(base_dir: str) -> tuple[dict, dict]:
    """
    Function that will extract all registered `.npy` files that are present in the dataset. We are assuming that the dataset has two directories at the root, one for before exposure and one for after exposure. If there have been multiple exposures, this will have to be modified.

    A sample dataset would have the following file structure:

        dataset_name/
            before_exposure/
                ...
            after_exposure/
                ...
            before_after_csv_pairs.csv <--- Output of this program
            ... <--- Other non directory files

    Args:
        base_dir (str): Root of the dataset
    Returns:
        Tuple with two dictionaries, one for each exposure. For each sample, the dictionary contains paths to all relevant registered files.
    """

    subdir_names = sorted(next(os.walk(base_dir))[1])
    assert len(subdir_names) == 2, "There should only be 2 subdirectories in the root"

    after_exposure_directory = subdir_names[0]
    before_exposure_directory = subdir_names[1]

    after_exposure_path = os.path.join(base_dir, after_exposure_directory)
    before_exposure_path = os.path.join(base_dir, before_exposure_directory)

    after_samples = find_npy_files(after_exposure_path, base_dir)
    before_samples = find_npy_files(before_exposure_path, base_dir)

    debug_print = False
    if debug_print:
        pprint_dict(before_samples, "Before samples")
        print()
        pprint_dict(after_samples, "After samples")

    return before_samples, after_samples


def extract_video_id(path: str) -> int:
    """
    Function for extracting the video ID from the `.mp4` videos. The expected input for this function would look like this: a/b/c/GXnnnnnn/GXnnnnnn_partX_processed_registered_stack_LIGHTGLUE.npy, where n is an integer in range [0,9].

    We only care about the `nnnnnn` part, the reason for this is that odd numbers correspond to the fron angle illumination while even numbers correspond to side angle illumination.

    Args:
        path (str): the path that we want to extract the video number from
    Returns:
        video ID of the path.
    """
    # We are matching on the /GXnnnnnn/ part as it allows us to rename the .npy files to something
    # more reasonable in the future if we wish to
    folder = path.split("/")[-2]
    match = re.search(r"(\d+)$", folder)

    if not match:
        raise ValueError(f"could not extract video id from {path}")

    return int(match.group(1))


def create_csv_pairs(
    before_paths: dict[str, list[str]], after_paths: dict[str, list[str]]
) -> list[tuple[str]]:
    """
    This function creates the csv pairs where each row represents one sample, and the columns represent the before and after exposure. The content of each cell is the path to the registered `.npy` file. The angles of illumination are also paired accordingly.

    Args:
        before_paths (dict[str, list[str]]): Paths to the `.npy` files before the exposure.
        after_paths (dict[str, list[str]]): Paths to the `.npy` files after the exposure.
    Returns:
        A list of pairs where each pair contains paths to the before/after registration of the sample.
    """

    # We only care about the cases where we have the registration for before and after exposure
    sample_intersections = before_paths.keys() & after_paths.keys()

    pairs = []
    for sample_name in sample_intersections:
        after_files = after_paths[sample_name]
        before_files = before_paths[sample_name]

        grouped_before = defaultdict(list)
        grouped_after = defaultdict(list)

        for path in before_files:
            video_id = extract_video_id(path)
            grouped_before[video_id % 2].append(path)

        for path in after_files:
            video_id = extract_video_id(path)
            grouped_after[video_id % 2].append(path)

        for parity in (0, 1):
            before_group = grouped_before[parity]
            after_group = grouped_after[parity]

            count = min(len(before_group), len(after_group))
            for i in range(count):
                pairs.append((before_group[i], after_group[i]))

    return pairs


def write_out_csv_file(root_directory: str, pairs_to_write: list[tuple[str]]) -> None:
    """
    Writes out the pairs into a csv file. Each pair represents one sample that has been registered before and after exposure, we also match the angle of illumination with the pairs. The header of the csv file determines which column contains the before exposure and after exposure files.

    An example output file would look something like this:
        Sample before exposure,Sample after exposure
        before_exp/1A1/front/GX013/GX011406_registered.npy,after_exp/1A1/front/GX011/GX011600_registered.npy
        ...

    Args: 
        root_directory (str): This is where the csv will be written
        pairs_to_write (list[tuple[str]]): The pairs that we want to write into the csv
    Returns:
        None
    """

    filename = "before_after_pairs.csv"
    out_file = os.path.join(root_directory, filename)
    print(out_file)

    with open(out_file, "w+", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Sample before exposure", "Sample after exposure"])
        writer.writerows(pairs_to_write)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    args = parse_args()

    before_paths, after_paths = match_before_and_after_npy_files(args.input)
    pairs = create_csv_pairs(before_paths, after_paths)
    write_out_csv_file(args.input, pairs)


if __name__ == "__main__":
    main()
