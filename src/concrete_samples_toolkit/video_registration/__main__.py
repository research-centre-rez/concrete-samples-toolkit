import os
import sys
import jsonschema
import argparse
import logging

from ..utils import pprint, load_config, load_json_schema
from ..utils.filename_builder import create_out_filename

from . import RegMethod, VideoRegistrator


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for registering a single video. Stores the registered video as a numpy matrix where each row represents one frame of the video. Default config can be found in video_registration/default_config.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argparser._action_groups.pop()
    required = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    # Required arguments
    required.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        required=True,
        help="Relative (or absolute) path to the video you want to register. Example: --i ../data/1/1-part0_processed.mp4",
    )
    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want the registered matrix to be saved. If 'auto' is passed, the program will automatically construct the name of the output file.\n"
            "  Example #1: --o auto -> will automatically append _registered_stack to the output file \n"
            "  Example #2: --o ../data/tmp_registered will save the registered matrix as ../data/tmp_registered.npy"
        ),
    )
    required.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["orb", "lightglue", "mudic"],
        required=True,
        help=(
            "Which method do you want to use for performing video registration. Options are:\n"
            "  orb: Performs ORB finding and matching of keypoints.\n"
            "  mudic: Performs digital image correlation for video registration.\n"
            "  lightglue: Performs registration with the use of the LightGlue neural network."
        ),
    )

    optional.add_argument(
        "-c",
        "--config",
        default= os.path.join(os.path.dirname(__file__), 'default_config.json5'),
        type=str,
        help="Path to a JSON5 config file that follows the config schema for video registration.",
    )

    return argparser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    # Console handler (shows INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))

    # File handler (only logs errors)
    fh = logging.FileHandler("failures.log")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    pprint.log_argparse(args)

    CONFIG_DIRECTORY = os.path.dirname(__file__)
    CONFIG_SCHEMA = load_json_schema(
        os.path.join(CONFIG_DIRECTORY, 'video_registration_schema.json')
    )
    try:
        config = load_config(args.config)
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        logger.info("Successfully validated the submitted config")
        pprint.pprint_dict(config, "Config parameters")
    except jsonschema.ValidationError as e:
        logger.error("Invalid configuration: \n %s", e.message)
        sys.exit(1)

    args.method = RegMethod[args.method.upper()]

    reg = VideoRegistrator(method=args.method, config=config)

    for input_path in args.input:
        if args.output in ["auto", "automatic"]:
            base, _ = os.path.splitext(input_path)
            output_path = create_out_filename(
                base, [], ["registered", "stack", args.method.name]
            )
        else:
            output_path = args.output

        try:
            reg_analysis = reg.get_registered_block(input_path)

            reg.save_registered_block(reg_analysis, output_path)

        except Exception as e:
            logger.error("%s", e)
            logger.error("%s", input_path)
            continue


if __name__ == "__main__":
    main()
