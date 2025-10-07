import logging
from typing import Optional
import cv2 as cv
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_video_matrix(vid_path: str, grayscale: bool, max_gb_memory: Optional[float] = None) -> np.ndarray:
    """
    This function decodes a whole video into a matrix where each row is a frame in the video. Warning, this can take a lot of space on the RAM. For a 1080p pre-processed video that contains ~660 frames it takes around 1.5GB memory.

    Parameters:
        `vid_path` (str): Path to the video file, if the path points to a non-existing file, returns an empty np array.
        `grayscale` (bool): Whether each frame is a single channel grayscale image.
        `max_gb_memory` (float): Optional parameter that sets a limit on memory usage. If the `video_matrix` would take more, returns zeroes.

    Returns:
        `frames` (np.ndarray): Video loaded into a matrix.
    """

    # First, check whether the video has been loaded
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        logger.error("Could not load video, returning")
        return np.zeros(0)

    # Estimate the memory requirement
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    channels = 1 if grayscale else 3
    bytes_per_pixel = 1  # Assuming uint8 image

    approx_bytes = frame_count * frame_height * frame_width * channels * bytes_per_pixel
    approx_gb = approx_bytes / (1024**3)

    logger.warning(
        "\n Decoding video to RAM — estimated memory usage: %s GB\n",
        round(approx_gb, 2),
    )

    if max_gb_memory is not None and approx_gb > max_gb_memory:
        logger.error("Estimated memory usage exceeds the maximum allowed memory usage. Returning 0s")
        return np.zeros(0)

    # Append the frames to a Python list first
    frames = []

    with tqdm(
        desc="Creating video matrix", total=cap.get(cv.CAP_PROP_FRAME_COUNT)
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if ret:
                if grayscale:
                    frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
                else:
                    frames.append(frame)
                pbar.update(1)
            else:
                break

    frames_np = np.array(frames)

    return frames_np
