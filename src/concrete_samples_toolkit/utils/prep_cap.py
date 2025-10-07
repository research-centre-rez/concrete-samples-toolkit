import sys
import logging
import cv2 as cv

def prep_cap(video_path:str, set_to=0) -> cv.VideoCapture:
    '''
    Helper function that will set the video capture to start on frame `set_to`. This is because we want to discard first `n` frames of a video due to image balancing noise.
    '''
    logger = logging.getLogger(__name__)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error: Could not open file %s", video_path)
        sys.exit(-1)

    # Skipping first 15 to account for whitening noise
    cap.set(cv.CAP_PROP_POS_FRAMES, set_to)
    return cap
