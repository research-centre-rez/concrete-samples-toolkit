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


def find_npy_files(exp_subdir, base_dir):
    return_dict = {}
    for current_dir, subdirectories, _ in os.walk(exp_subdir):
        if "front" in subdirectories:
            sample_name = current_dir.split("/")[-1]
            for face_subdir in subdirectories:
                # Used for indexing
                sample_face = os.path.join(sample_name, face_subdir)
                return_dict[sample_face] = []

                sample_subdir = os.path.join(current_dir, face_subdir)

                for current_face_dir, _, face_files in os.walk(sample_subdir):
                    out_dir = current_face_dir.replace(base_dir, "")
                    npy_files = [
                        os.path.join(out_dir, npy_file)
                        for npy_file in face_files
                        if ".npy" in npy_file.lower()
                    ]
                    if npy_files:
                        return_dict[sample_face].extend(npy_files)

    return return_dict


def collect_paths(base_dir):

    subdir_names = sorted(os.listdir(base_dir))
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
    folder = path.split("/")[-2]
    match = re.search(r"(\d+)$", folder)
    if not match:
        raise ValueError(f"could not extract video id from {path}")
    return int(match.group(1))


def create_csv_pairs(
    before_paths: dict[str, list[str]], after_paths: dict[str, list[str]]
) -> list[tuple[str]]:
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

    before_paths, after_paths = collect_paths(args.input)
    pairs = create_csv_pairs(before_paths, after_paths)
    write_out_csv_file(args.input, pairs)


if __name__ == "__main__":
    main()
