import os
import sys
import argparse
import logging

import jsonschema
from jsonschema.exceptions import ValidationError

from ..utils import pprint, load_json_schema, load_config
from ..utils.filename_builder import append_file_extension, create_out_filename
from . import VideoProcessor
from . import ProcessorMethod


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program that is used for performing rotation correction on a given video. The new video will be saved separately.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    # Required arguments
    req.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Relative (or absolute) path to the video you want to process. Example: --i ../data/1/1-part0.mp4",
    )
    req.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want the video to be saved. If 'auto' is passed, the program will automatically construct the name of the output file.\n"
            " Example #1: --o auto -> will automatically append '-processed' to the input file and save it like that.\n"
            " Example #2: --o ../data/tmp.mp4 -> store the video as 'tmp.mp4' at the location '../data/'."
        ),
    )
    req.add_argument(
        "-m",
        "--method",
        choices=["none", "opt_flow", "approx"],
        required=True,
        help=(
            "Which video processing method to use? \n"
            " none - No rotation correction, mainly used for other types of video processing. \n"
            " opt_flow - Optical flow with the parameters from the config file to undo rotation. \n"
            " approx - Pre-calculated rotation center and angle correction to undo rotation. \n"
            "For each of these methods, you can downscale, downsample and grayscale the resulting video."
        ),
    )
    optional.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), 'default_config.json5'),
        type=str,
        help="Path to a JSON config file that follows the config schema for video processing. The default config schema can be found in './video_process/default_config.json5'",
    )

    return argparser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    CONFIG_DIRECTORY = os.path.dirname(__file__)
    CONFIG_SCHEMA = load_json_schema(os.path.join(CONFIG_DIRECTORY, "video_processing_schema.json"))
    try:
        config = load_config(args.config)
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        logger.info("Successfully loaded a JSON schema")
        pprint.pprint_dict(config, desc="Config parameters")
    except ValidationError as e:
        logger.error("Invalid configuration: \n %s", e.message)
        sys.exit(1)

    proc = VideoProcessor(method=ProcessorMethod[args.method.upper()], config=config)
    for input_path in args.input:

        if args.output in ["auto", "automatic"]:
            base, _ = os.path.splitext(input_path)
            save_as = create_out_filename(base, [], ["processed"])
            output_path = append_file_extension(save_as, "mp4")
        else:
            output_path = args.output

        logger.info("Processing %s -> %s", input_path, output_path)
        analysis = proc.get_rotation_analysis(input_path)
        proc.write_out_video(input_path, analysis, output_path)
        logger.info("==================================================")


if __name__ == "__main__":
    main()
