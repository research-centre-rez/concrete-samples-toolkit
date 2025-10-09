import argparse
import logging
import os
import sys

from . import ImageFuserFactory, FuseMethod, crop_image
from ..utils import pprint
from ..utils.filename_builder import create_out_filename


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for image fusion of a video stack. The goal of the program is to take a (registered) video stack and fuse the individual frames of the stack into a single image. Available fusing methods are listed below in `--methods`",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Relative (or absolute) path to a registered video stack. You can pass in multiple files at the same time. Each stack is fused individually, i.e. for one registered stack there will be one set of fused images.",
    )

    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want to save the fused image(s), you can set the extension to be either `.jpg` or `.png`, if you don't pass a file extension then the program will warn you but automatically append `.png` extension to the filename. If 'auto' is passed, the program will automatically construct the name of the output file.\n"
            " Example: --i ./filename --o auto -> ./filename_fused"
            "Each method will also append its name to the end of the file (when saving to disc)."
            " Example: --i ./filename --o auto --m max -> ./filename_fused_MAX.png"
            "NOTE: If you are passing in multiple files, always use --o auto (otherwise the new files will always be rewritten)"
        ),
    )

    required.add_argument(
        "-m",
        "--method",
        nargs="+",
        required=True,
        choices=["min", "max", "var", "med", "mean", "all"],
        help=(
            "Which image fusing methods do you want to use? The choices are:"
            "\n min: Take minimum from each frame across the whole stack"
            "\n max: Take maximum from each frame across the whole stack"
            "\n var: Variance of each pixel across the stack"
            "\n med: Median of each pixel across the stack"
            "\n mean: Mean of each pixel across the stack"
            "\n all: Use all of the available methods"
        ),
    )

    optional.add_argument(
        "--crop", 
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Should the final image be cropped such that it only contains the sample? Background will be thrown away."
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    fuser_factory = ImageFuserFactory()

    # Using all fusing methods
    if "all" in args.method:
        methods = [FuseMethod[e.name] for e in FuseMethod]

    # Using only the selected fusing methods
    else:
        methods = [FuseMethod[m.upper()] for m in args.method]

    # Getting all of the required fusers
    fusers = [fuser_factory.get_fuser(method) for method in methods]

    # If cropping is required, find the MIN fuser
    min_fuser = None
    if args.crop:
        for f in fusers:
            if f.method == FuseMethod.MIN:
                min_fuser = f
                break
        if min_fuser is None:
            min_fuser = fuser_factory.get_fuser(FuseMethod.MIN)

    gallery = {}

    # Go over each of the input files
    for input_path in args.input:
        # Create the automatic filename
        if args.output in ["auto", "automatic"]:
            path, name = os.path.split(input_path)
            base, _ = os.path.splitext(name)
            output_path = create_out_filename(
                os.path.join(path, "images", base), [], ["fused"]
            )
        else:
            output_path = args.output

        logger.info("Processing %s -> %s", input_path, output_path)

        for fuser in fusers:
            # Create a gallery for later image processing
            gallery[fuser.method] = fuser.get_fused_image(input_path)

            if args.crop:
                min_mask = min_fuser.get_min_mask(input_path)

                for method, image in gallery.items():
                    gallery[method] = crop_image(image, min_mask)

            # Save the image to disc
            fuser.save_image_to_disc(gallery[fuser.method], output_path)


if __name__ == "__main__":
    main()
