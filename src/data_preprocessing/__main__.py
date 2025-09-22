"""
Video preprocessing

OpenCV has a default setting that specifies how many frames it will read from a video, you should change this before running any of these functions with:

export OPENCV_FFMPEG_READ_ATTEMPTS=100000

While the os.environ written below should do it for you, it is possible that it might not work.
"""
import os
import argparse
import logging
import shutil

#from utils import pprint
from . import detect_black_frames
from . import split_video

from utils import pprint

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100000"
# TODO: Add this to docker

logger = logging.getLogger(__name__)


def parse_args():
    argparser = argparse.ArgumentParser(description='Program for identifying black sections of a video and splitting the video into 2 separate parts. Note: The parts are not labeled according to their light angle, they are arbitrarily numbered.')

    req = argparser.add_argument_group('required arguments')

    # Required arguments
    req.add_argument('-i', '--input', type=str, required=True, help='Path to the input video or to the directory that contains videos that you want to split. The program will run through all of the videos that it can find in the directory.')
    req.add_argument('-o', '--output', type=str,required=True, help='Where do you want the videos to be saved')

    return argparser.parse_args()

def _split_videos_in_directory(directory_path:str, output_dir:str) -> None:
    '''
    Processes all videos found in `directory_path` and saves them in `output_dir`.
    '''
    os.makedirs(output_dir, exist_ok=True)

    found_any = False
    for root, _, files in os.walk(directory_path):
        for file in sorted(files):
            if file.upper().endswith('.MP4'):
                rel_dir = os.path.relpath(root, directory_path)
                rel_out_dir = os.path.join(output_dir, rel_dir)
                video_path = os.path.join(root, file)

                _split_video(video_path, rel_out_dir)
                found_any = True

    # In case no videos were processed successfully
    if not found_any or not os.listdir(output_dir):
        try:
            os.rmdir(output_dir)
        except OSError:
            pass  # Directory not empty or failed for some other reason

def _split_video(vid_path:str, out_path:str) -> None:
    os.makedirs(out_path, exist_ok=True)
    try:
        black_f_id, fps = detect_black_frames(vid_path)
        stats           = split_video(vid_path, black_f_id, out_path, fps)

        logger.info('Succesfully split a video. With the following stats:')
        for key, value in stats.items():
            logger.info('%s has length %s seconds', key, value)

    except:
        logger.error("Wasn't able to split video: %s. Removing directory %s", vid_path, out_path)

        # Directory is empty
        if os.listdir(out_path):
            os.rmdir(out_path)

        # Do nothing if dir non-empty since other (successfully created)
        # videos could already be there. We don't want to remove those


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    pprint.log_argparse(args)

    if os.path.isdir(args.input):
        logger.info('Processing all video in directory %s', args.input)
        _split_videos_in_directory(directory_path=args.input, output_dir=args.output)

    else:
        logger.info('Processing video %s', args.input)
        _split_video(vid_path=args.input, out_path=args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
