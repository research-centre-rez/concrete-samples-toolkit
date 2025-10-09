import os
import logging
import sys
import cv2 as cv
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def monte_carlo_mean(image: np.ndarray, sample_size: int = 10000, seed: int = None) -> float:
    """
    Estimate the mean pixel intensity of a 2D image using Monte Carlo sampling.

    Parameters:
    - image: 2D NumPy array representing the image.
    - sample_size: Number of pixels to sample. Default is 10,000.
    - seed: Random seed for reproducibility.

    Returns:
    - Estimated mean pixel intensity.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Get the total number of pixels
    total_pixels = image.size

    # Flatten the image for efficient random access
    flat_image = image.ravel()

    # Choose random indices
    sample_indices = np.random.randint(0, total_pixels, size=sample_size)

    # Sample the pixel values and compute the mean
    sample_values = flat_image[sample_indices]
    estimated_mean = np.mean(sample_values)

    return estimated_mean


def grid_sample_mean(image:np.ndarray, step:int=100) -> float:
    sampled_pixels = image[::step, ::step]
    return np.mean(sampled_pixels)


def detect_black_frames(vid_path:str, threshold=5):
    """
    Given a video path, identify the frames where there the screen turns black
    Args:
        vid_path (str): Path to the video where you want to find black parts
        threshold (int): Maximum threshold for a frame to be considered not black
    Returns:
        black_frames_ids:list
        fps: float
    """

    prev_black = True

    cap = cv.VideoCapture(vid_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    if cap.isOpened():
        logger.info('Succesfully loaded video: %s', vid_path)

    black_frame_indices = []
    frame_count         = int(cap.get((cv.CAP_PROP_FRAME_COUNT)))

    with tqdm(range(frame_count), leave=False) as progress:
        for i in progress:
            ret, frame = cap.read()

            if not ret:
                break

            mean = grid_sample_mean(frame,3000)
            #mean = monte_carlo_mean(frame, 100)
            progress.set_postfix(mean=mean)

            # End of video and its not black
            if i == frame_count-1 and not prev_black:
                black_frame_indices.append(i)

            # Previous frame was black and the current one is not
            elif prev_black and mean > threshold:
                black_frame_indices.append(i)
                prev_black = False

            # Previous frame wasn't black and the current one is
            elif not prev_black and mean < threshold:
                black_frame_indices.append(i-1)
                prev_black = True

    cap.release()
    logger.info("Indices where there are black frames: %s", black_frame_indices)
    return black_frame_indices, fps
